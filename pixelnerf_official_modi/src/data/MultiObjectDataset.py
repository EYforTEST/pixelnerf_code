import os
import glob
import json
import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

import cv2 as cv2

from util import get_image_to_tensor_balanced, get_mask_to_tensor# , get_depth_to_tensor


class MultiObjectDataset(torch.utils.data.Dataset):
    """Synthetic dataset of scenes with multiple Shapenet objects"""
    def __init__(self, path, stage="train", z_near=0.9, z_far=1.4, n_views=None):
        super().__init__()
        path = os.path.join(path, stage)

        self.base_path = path
        print("Loading NeRF synthetic dataset", self.base_path) # ../human\train

        trans_files = []
        
        TRANS_FILE = "transforms_train.json"
        
        for root, directories, filenames in os.walk(self.base_path):
            # print(filenames)

            if TRANS_FILE in filenames:
                trans_files.append(os.path.join(root, TRANS_FILE)) # ../human/train/004_00/transforms_train.json, ../human/val/010_00/transforms_train.json

        self.trans_files = trans_files

        self.image_to_tensor = get_image_to_tensor_balanced()
        # self.depth_to_tensor = get_depth_to_tensor()
        self.mask_to_tensor = get_mask_to_tensor()

        self.z_near = z_near
        self.z_far = z_far

        self.lindisp = False
        self.n_views = n_views

    def __len__(self):
        return len(self.trans_files)

    def _check_valid(self, index):
        if self.n_views is None:
            return True
        trans_file = self.trans_files[index]
        dir_path = os.path.dirname(trans_file)

        try:
            with open(trans_file, "r") as f:
                transform = json.load(f)
        except Exception as e:
            print("Problematic transforms.json file", trans_file)
            print("JSON loading exception", e)
            return False
        
        if len(transform["frames"]) != self.n_views:
            return False
        
        if len(glob.glob(os.path.join(dir_path, "*.png"))) != self.n_views:
            return False
        
        return True

    def __getitem__(self, index):
        if not self._check_valid(index):
            return {}

        trans_file = self.trans_files[index]
        dir_path = os.path.dirname(trans_file) # ../human\train\038_00

        print(dir_path)
 
        with open(trans_file, "r") as f: transform = json.load(f)

        # file_paths = os.listdir(os.path.join(dir_path, "images"))

        all_imgs = []
        all_src_imgs = []

        all_bboxes = []
        all_src_bboxes = []

        all_poses = []
        all_src_poses = []
        
        for ordering, frame in enumerate(transform["frames"]):
            fpath = frame["file_path"] # ./images/006_00_00
  
            basename = os.path.splitext(os.path.basename(fpath))[0] # 038_00_00
            
            obj_img_path = os.path.join(dir_path, "images/{}.jpg".format(basename)) # ../human\train\038_00\out/dense/0/images/038_00_00.jpg
            obj_dep_path = os.path.join(dir_path, "dense_depth/{}.npy".format(basename))

            if os.path.isfile(obj_img_path) and os.path.isfile(obj_dep_path):
                img = imageio.imread(obj_img_path)

                if ordering == 0:
                    ORI_H, ORI_W = img.shape[0], img.shape[1]
                    Resize_H, Resize_W = ORI_H // 7, ORI_W // 7

                    # Resize_H, Resize_W = img.shape[0], img.shape[1]
                
                img = cv2.resize(img, dsize=(Resize_W, Resize_H))

                # depth = cv2.resize(depth[0], dsize=(Resize_W, Resize_H)) 
                # depth = np.load(obj_dep_path)

                # depth = (depth - depth.min()) / (depth.max() - depth.min()) # 0-1

                ######################################################
                rows = np.any(img, axis=1)
                cols = np.any(img, axis=0)

                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
        
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]

                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

                ######################################################
                
                img_tensor = self.image_to_tensor(img[..., :3])
                # depth_tensor = self.depth_to_tensor(depth)

                ######################################################
                
                '''
                D:/eyyou/human/train\014_00\images
                ['014_00_10.jpg', '014_00_17.jpg']
                D:/eyyou/human/train\016_00\images
                ['016_00_10.jpg', '016_00_17.jpg']
                D:/eyyou/human/train\020_00\images
                ['020_00_10.jpg', '020_00_17.jpg']
                D:/eyyou/human/train\025_00\images
                ['025_00_10.jpg', '025_00_17.jpg']
                D:/eyyou/human/train\042_00\images
                ['042_00_10.jpg', '042_00_17.jpg']
                '''

                if basename.split('_')[0] in ['014', '016', '020', '025', '042']:
                    if basename.split('_')[-1] in ['03','10','17']:
                        # all_src_depth.append(depth_tensor)
                        all_src_imgs.append(img_tensor)

                        all_src_bboxes.append(bbox)
                        all_src_poses.append(torch.tensor(frame["transform_matrix"]))
                    else:
                        # all_depth.append(depth_tensor)
                        all_imgs.append(img_tensor)

                        all_bboxes.append(bbox)
                        all_poses.append(torch.tensor(frame["transform_matrix"]))
                else:
                    if basename.split('_')[-1] in ['00','10','17']:
                        # all_src_depth.append(depth_tensor)
                        all_src_imgs.append(img_tensor)

                        all_src_bboxes.append(bbox)
                        all_src_poses.append(torch.tensor(frame["transform_matrix"]))
                    else:
                        # all_depth.append(depth_tensor)
                        all_imgs.append(img_tensor)

                        all_bboxes.append(bbox)
                        all_poses.append(torch.tensor(frame["transform_matrix"]))
            else:
                continue

        imgs = torch.stack(all_imgs)
        bboxes = torch.stack(all_bboxes)
        # depths = torch.stack(all_depth)
        poses = torch.stack(all_poses)

        src_imgs = torch.stack(all_src_imgs)
        src_bboxes = torch.stack(all_src_bboxes)
        # src_depths = torch.stack(all_src_depth)
        src_poses = torch.stack(all_src_poses)

        H, W = imgs.shape[-2:]
        camera_angle_x = transform.get("camera_angle_x")
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        # focal = sum(all_src_focal) / len(all_src_focal)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            # "depth": depths, 
            "images": imgs,
            "bbox": bboxes,
            "poses": poses,

            # "src_depth": src_depths, 
            "src_images": src_imgs,
            "src_bbox": src_bboxes,
            "src_poses": src_poses,
        }

        return result