import sys
import os
import numpy as np
import torch
from dotmap import DotMap
import matplotlib.pyplot as plt
import imageio
import tqdm 
import warnings
from PIL import Image
import random

from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.functional import to_tensor

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pytorch_ssim import SSIM
from skimage.metrics import structural_similarity as ssim 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)   

def seed_fix(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    return

def extra_args(parser):
    parser.add_argument(
        "--batch_size", 
        "-B", 
        type=int, 
        default=1, 
        help="Object batch size ('SB')"
    )
    
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )

    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=0,
        help="Step to stop using bbox sampling",
    )

    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--cuda",
        default='0',
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
    )

    return parser

seed_fix(100)
args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir) 
print("dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp))

net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc

if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp,).to(device=device)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()
nviews = list(map(int, args.nviews.split()))

class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)

        self.renderer_state_path = f"{self.args.checkpoints_path}/{self.args.name}/_renderer"

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print("lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine))

        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]

        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]

        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(torch.load(self.renderer_state_path, map_location=device))

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0

        self.psnr_loss_path = os.path.join(self.args.visual_path, self.args.name)
        self.train_loss_plot, self.eval_loss_plot, self.val_psnr_plot = [], [], []
        self.val_g_ssim_plot, self.val_rgb_ssim_plot, self.val_ms_ssim_plot = [], [], []

        self.ssim = SSIM(window_size=11)
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True)
    
    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train, global_step=0):
        if "images" not in data:
            return {}
        
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)

        SB, NV, _, H, W = all_images.shape

        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax

        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)

        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1: image_ord = torch.randint(0, NV, (SB, 1))
        else: image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]

            c = None
            if "c" in data: c = data["c"][obj_idx]

            if curr_nviews > 1: image_ord[obj_idx] = torch.from_numpy(np.random.choice(NV, curr_nviews, replace=False))
            
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)  # (NV, H, W, 8)
            
            rgb_gt_all = (images_0to1.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)) # (NV, H, W, 3) rgb_gt_all: torch.Size([5880000 = 49*300*400, 3])
            
            ############### 한 물체 내의 모든 pixel (viewdir. 수 * H * W) 중 몇 개의 pixel만 선택해서 학습 진행 ###############
            
            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
            else:
                pix_inds = torch.randint(0, NV * H * W, (args.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]  # (ray_batch_size, 3) rgb_gt: torch.Size([128, 3])
              
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(device=device)  # (ray_batch_size, 8) rays: torch.Size([128, 8]
            
            all_rgb_gt.append(rgb_gt)           
            all_rays.append(rays)

        ############ 하나의 물체로만 한 번의 학습 진행 ##############
        
        all_rgb_gt = torch.stack(all_rgb_gt) # rgb_gt_for_patch.reshape(1,-1,3)  # (SB, ray_batch_size, 3) : (1, 200, 3)
        all_rays = torch.stack(all_rays) # rays_for_patch.reshape(1,-1,8)  # (SB, ray_batch_size, 8) : (1, 200, 8)

        ############ 하나의 물체 중 N개의 VIEW 활용 feature 추출 ##############

        image_ord = image_ord.to(device)

        src_images = util.batched_index_select_nd(all_images * 0.5 + 0.5, image_ord)  # (SB, NS, 3, H, W)
        src_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        # for fixed src views
        # src_images = (data["src_images"] * 0.5 + 0.5).to(device=device)  # (SB, NS, 3, H, W)
        # src_poses = data["src_poses"].to(device=device)  # (SB, NS, 4, 4)
        
        all_bboxes = all_poses = all_images = None
        
        # models.py PixelNeRFNet 중 encode 로 이동 #
        ############## source view 로부터 featrue extract 하는 부분 ##############
        net.encode(
            # src_depths, 
            src_images,
            src_poses,
            all_focals.to(device=device),
            c=all_c.to(device=device) if all_c is not None else None,
        )

        render_dict = DotMap(render_par(all_rays, want_weights=True,))

        coarse = render_dict.coarse
        fine = render_dict.fine

        loss_dict = {}
        
        print(f'rgb: {all_rgb_gt.mean().item()} {coarse.rgb.mean().item()} {fine.rgb.mean().item()}') 
 
        # coarse
        coarse_rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = coarse_rgb_loss.item()

        # fine
        fine_rgb_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
        loss_dict["rf"] = fine_rgb_loss.item()

        # total
        rgb_loss = coarse_rgb_loss + fine_rgb_loss 

        loss = rgb_loss 

        if is_train: loss.backward()
        loss_dict["t"] = loss.item()

        if is_train == True:
            self.train_loss_plot.append(loss.item())
            
            plt.figure()
            plt.plot(self.train_loss_plot)
            plt.title(f'train_loss | min_one: {self.train_loss_plot.index(min(self.train_loss_plot))} iter_{min(self.train_loss_plot)}')

            plt.savefig(os.path.join(self.visual_path,"train_loss_plot.png"))
            plt.close()
        else:
            self.eval_loss_plot.append(loss.item())

            plt.figure()
            plt.plot(self.eval_loss_plot)
            plt.title(f'eval_loss | min_one: {self.eval_loss_plot.index(min(self.eval_loss_plot))} iter_{min(self.eval_loss_plot)}')

            plt.savefig(os.path.join(self.visual_path,"eval_loss_plot.png"))
            plt.close()

        return loss_dict

    def train_step(self, data, global_step):
        renderer.train()
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()       
        return self.calc_losses(data, is_train=False, global_step=global_step)

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx

        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        src_images = data["src_images"][batch_idx].to(device=device)  # (NV, 3, H, W)

        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        src_poses = data["src_poses"][batch_idx].to(device=device)  # (NV, 4, 4)

        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")

        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)

        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)

        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
        src_images_0to1 = src_images * 0.5 + 0.5  # (NV, 3, H, W)

        view_dest = 6 

        # set renderer net to eval mode
        renderer.eval()

        source_views = src_images_0to1.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, H, W, 3) # (NV, H, W, 3)

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)

        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = src_images_0to1  # (NS, 3, H, W)

            net.encode(
                test_images.unsqueeze(0),
                src_poses.unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )

            test_rays = test_rays.reshape(1, H * W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)

            if using_fine:
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb[0].cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print("c alpha min {}, max {}".format(alpha_coarse_np.min(), alpha_coarse_np.max()))
        print("c depth min {}, max {}".format(depth_coarse_np.min(), depth_coarse_np.max()))

        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print("f alpha min {}, max {}".format(alpha_fine_np.min(), alpha_fine_np.max()))
            print("f depth min {}, max {}".format(depth_fine_np.min(), depth_fine_np.max()))

            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255

            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]

            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))

            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        # psnr
        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # gray scale ssim, ms ssim
        recon_image = to_tensor(rgb_to_grayscale(Image.fromarray((rgb_fine_np * 255).astype(np.uint8)))).unsqueeze(0)
        target_image = to_tensor(rgb_to_grayscale(Image.fromarray((gt * 255).astype(np.uint8)))).unsqueeze(0)

        gray_ssim = self.ssim(recon_image, target_image)
        ssim_vals = {"ssim": gray_ssim}
        print("gray_ssim", gray_ssim)     

        # grb scale ssim
        recon_image = to_tensor(rgb_fine_np).unsqueeze(0)
        target_image = to_tensor(gt).unsqueeze(0)

        rgb_ssim = self.ssim(recon_image, target_image)
        print("rgb_ssim", rgb_ssim)       

        ms_ssim = self.ms_ssim(recon_image, target_image)
        ms_ssim_vals = {"msssim": ms_ssim}
        print("ms_ssim", ms_ssim)   
        
        # PLOT
        self.val_psnr_plot.append(psnr)

        self.val_g_ssim_plot.append(gray_ssim)
        self.val_ms_ssim_plot.append(ms_ssim)
        self.val_rgb_ssim_plot.append(rgb_ssim)

        plt.figure()
        plt.plot(self.val_psnr_plot)
        plt.title(f'psnr | best_one: {self.val_psnr_plot.index(max(self.val_psnr_plot))} iter_ {max(self.val_psnr_plot)}')
        plt.savefig(os.path.join(self.visual_path,"vis_psnr_plot.png"))
        plt.close()
        plt.switch_backend('agg')
        plt.close('all')

        plt.figure()
        plt.plot(self.val_g_ssim_plot)
        plt.title(f'ssim | best_one: {self.val_g_ssim_plot.index(max(self.val_g_ssim_plot))} iter_ {max(self.val_g_ssim_plot)}')
        plt.savefig(os.path.join(self.visual_path,"vis_gray_ssim_plot.png"))
        plt.close()
        plt.switch_backend('agg')
        plt.close('all')

        plt.figure()
        plt.plot(self.val_ms_ssim_plot)
        plt.title(f'ssim | best_one: {self.val_ms_ssim_plot.index(max(self.val_ms_ssim_plot))} iter_ {max(self.val_ms_ssim_plot)}')
        plt.savefig(os.path.join(self.visual_path,"vis_ms_ssim_plot.png"))
        plt.close()
        plt.switch_backend('agg')
        plt.close('all')

        plt.figure()
        plt.plot(self.val_rgb_ssim_plot)
        plt.title(f'ssim | best_one: {self.val_rgb_ssim_plot.index(max(self.val_rgb_ssim_plot))} iter_ {max(self.val_rgb_ssim_plot)}')
        plt.savefig(os.path.join(self.visual_path,"vis_rgb_ssim_plot.png"))
        plt.close()
        plt.switch_backend('agg')
        plt.close('all')

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals, ssim_vals, ms_ssim_vals
   
    def gen_video(self, data, global_step, idx=None):
        batch_idx = 0

        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        src_images = data["src_images"][batch_idx].to(device=device)  # (NV, 3, H, W)

        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        src_poses = data["src_poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)

        c = data.get("c")
        if c is not None: c = c[batch_idx : batch_idx + 1]  # (1)

        NV, _, H, W = images.shape

        cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)  # (NV, H, W, 8)

        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
        src_images = src_images * 0.5 + 0.5  # (NV, 3, H, W)

        # set renderer net to eval mode
        renderer.eval()
                
        all_rgb_coarse = []
        all_rgb_fine = []

        with torch.no_grad():
            test_images = src_images   # (NS, 3, H, W)

            net.encode(
                test_images.unsqueeze(0),
                src_poses.unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )

            for rays in tqdm.tqdm(torch.split(cam_rays.view(-1, 8), 6400, dim=0)):
                render_dict = DotMap(render_par(rays[None].to(device), want_weights=False))

                coarse = render_dict.coarse
                fine = render_dict.fine

                all_rgb_coarse.append(coarse.rgb[0])
                all_rgb_fine.append(fine.rgb[0])

        rgb_coarse = torch.cat(all_rgb_coarse)
        rgb_fine = torch.cat(all_rgb_fine)

        frames_coarse = rgb_coarse.view(-1, H, W, 3)
        frames_fine = rgb_fine.view(-1, H, W, 3)

        print("Writing video")
            
        vid_name = "_v"
        vid_co_path = os.path.join(self.visual_path, "video_coarse" + vid_name + ".gif")
        vid_fi_path = os.path.join(self.visual_path, "video_fine" + vid_name + ".gif")

        imageio.mimwrite(vid_co_path, (frames_coarse.cpu().numpy() * 255).astype(np.uint8), format='gif')
        imageio.mimwrite(vid_fi_path, (frames_fine.cpu().numpy() * 255).astype(np.uint8), format='gif')

        viewimg_path = os.path.join(self.visual_path, "video" + vid_name + "_view.jpg")
        
        img_np = (src_images.permute(0, 2, 3, 1)).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.hstack((*img_np,))

        imageio.imwrite(viewimg_path, img_np)

        for img_i in range(len(frames_fine)):
            img = frames_fine[img_i].cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())

            gt = images_0to1[img_i].permute(1,2,0).cpu().numpy()
            gt = (gt - gt.min()) / (gt.max() - gt.min())

            img_ = (img * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(self.visual_path, f"{img_i}_" + "video" + vid_name + "_view.jpg"), (img_))

            H,W = img.shape[:2]

            psnr_vals = util.psnr(img, gt)
            ssim_vals = (ssim(img[:,:,0], gt[:,:,0]) + ssim(img[:,:,1], gt[:,:,1]) + ssim(img[:,:,2], gt[:,:,2])) / 3

            ms_ssim_vals = self.ms_ssim(torch.Tensor(img.reshape(1,3,H,W)), torch.Tensor(gt.reshape(1,3,H,W)))
            lpips_vals = self.lpips(torch.Tensor(img.reshape(1,3,H,W)), torch.Tensor(gt.reshape(1,3,H,W)))

            print(f"{img_i} | psnr: {psnr_vals} ssim: {ssim_vals} msssim: {ms_ssim_vals} lpips: {lpips_vals}")

        print("Wrote to", vid_fi_path, "view:", viewimg_path)

        # set the renderer network back to train mode
        renderer.train()
        return 
    
trainer = PixelNeRFTrainer()
trainer.start()

