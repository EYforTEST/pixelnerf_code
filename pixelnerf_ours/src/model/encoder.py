"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import util
from model.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler

import numpy as np

from torch import Tensor
from typing import Optional
from torch.nn.functional import pad, avg_pool2d, grid_sample as tgrid_sample

def exponential_padding(img, padding: int, double_width):
    """
    implements padding strategy similar to border padding but exponentially increases border value with
    increasing distance to border.
    I.e: f(x) = f(x_border) * exp(ln(2) * (x-x_border) / double_width)
    Used for extrapolating depth standard deviation.
    :param img:
    :param padding:
    :param double_width:
    :return:
    """
    N, C, H, W = img.shape
    base = pad(img, [padding] * 4, mode="replicate")
    exponents = torch.zeros(*img.shape[:-2], H + 2 * padding, W + 2 * padding, dtype=img.dtype, device=img.device)

    for i in range(padding):
        idx = padding - (i + 1)
        exponents[:, :, idx, :] = i
        exponents[:, :, -(idx + 1), :] = i
        exponents[:, :, :, idx] = i
        exponents[:, :, :, -(idx + 1)] = i

    out = base * torch.exp(exponents / double_width * np.log(2))
    return out

def grid_sample(input: Tensor,
                grid: Tensor,
                mode: str = "bilinear",
                padding_mode: str = "zeros",
                align_corners: Optional[bool] = None,
                pad_double_width=20, pad_size=40, exp_padding_mode="border"
                ) -> Tensor:
    """
    extends pytorch grid sample by exponential padding
    Parameters
    ----------
    input
    grid
    mode
    padding_mode
    align_corners

    Returns
    -------

    """
    if padding_mode != "exponential":
        return tgrid_sample(input, grid, mode, padding_mode, align_corners)

    else:
        H, W = input.shape[-2:]
        img_size = torch.tensor([W, H], dtype=torch.float, device=input.device)
        input_padded = exponential_padding(input, pad_size, pad_double_width)

        # correcting grid
        if align_corners:  # -1 / +1 referring to outer pixel centers
            scale_factor = (img_size - 1) / (img_size + 2 * pad_size - 1)
        else:
            scale_factor = img_size / (img_size + 2 * pad_size)
        grid = grid * scale_factor.view(1, 1, 1, 2)
        return tgrid_sample(input_padded, grid, mode=mode, padding_mode=exp_padding_mode, align_corners=align_corners)

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "images encoder")
            self.model = getattr(torchvision.models, backbone)(pretrained=pretrained, norm_layer=norm_layer)
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

            # for depth feature
            print("Using torchvision", backbone, "depths encoder")
            self.model_depth = getattr(torchvision.models, backbone)(pretrained=pretrained, norm_layer=norm_layer)
            self.model_depth.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # Following 2 lines need to be uncommented for older configs
            self.model_depth.fc = nn.Sequential()
            self.model_depth.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer("latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False)
        # self.latent (B, L, H, W)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """

        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)

            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )

            return samples[:, :, :, 0]  # (B, C, N)

    def index_depth(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """

        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)

            samples = F.grid_sample(
                self.depths,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )

            return samples[:, :, :, 0]  # (B, C, N)

    def resnetout(self, x):
        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]

            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)

            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)

            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)

            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]

            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
                
            self.latent = torch.cat(latents, dim=1)

        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0

        return self.latent
    
    def resnetout_depth(self, x):
        if self.use_custom_resnet:
            latent = self.model_depth(x)
        else:
            x = self.model_depth.conv1(x)
            x = self.model_depth.bn1(x)
            x = self.model_depth.relu(x)

            latents = [x]

            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model_depth.maxpool(x)
                x = self.model_depth.layer1(x)
                latents.append(x)

            if self.num_layers > 2:
                x = self.model_depth.layer2(x)
                latents.append(x)

            if self.num_layers > 3:
                x = self.model_depth.layer3(x)
                latents.append(x)

            if self.num_layers > 4:
                x = self.model_depth.layer4(x)
                latents.append(x)

            latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]

            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
                
            latent = torch.cat(latents, dim=1)
        
        return latent

    def forward(self, depths, images):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        
        # print(images.shape) # torch.Size([3, 3, 585, 428])
        # print(depths.shape) # torch.Size([3, 1, 585, 428])

        images = images.to(device=self.latent.device) # print(x.shape) # torch.Size([3, 3, 585, 428])
        self.latent = self.resnetout(images) # print(self.latent.shape) # torch.Size([3, 512, 293, 214])

        depths = depths.to(device=self.latent.device) # print(x.shape) # torch.Size([3, 1, 585, 428])
        self.depths = self.resnetout_depth((depths)) # print(self.depths.shape)  # torch.Size([3, 512, 293, 214])      

        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )

class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)

        self.model.fc = nn.Sequential()

        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        print(f'################################### x shape: {x.shape}')

        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
