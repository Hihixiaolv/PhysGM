# Copyright (c) 2025, Zequn Chen.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kiui
from kiui.lpips import LPIPS

from easydict import EasyDict as edict
from einops import rearrange
from gsplat import rasterization
from PIL import Image

from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2
from .transformer import TransformerBlock
from .dpt import DPTHead
from .dense_rep_encoder import DenseRepresentationEncoder


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def make_transform(resize_size=256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

class DINOv3FeatureExtractor(nn.Module):

    def __init__(self, model_path, device):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.transform = make_transform()
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):

        inputs = self.transform(images)
        B, C, H, W = inputs.shape
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]  

        BV, num_patches, dim = patch_embeddings.shape
        h = H // 16
        w = W // 16
        features = patch_embeddings.reshape(BV, h, w, dim)
        
        return features

class Processor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.model.num_layers
        self.dim = config.model.dim
        self.blocks = nn.ModuleList()

        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(self.dim, config.model.transformer.head_dim))

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: B and D remain the same, L might change if there are merge layers
        """

        save_ids = [4, 11, 17, 23]
        aggregated_token_list = []

        for i in range(self.num_layers):
            x = self.blocks[i](x)
            if i in save_ids:
                aggregated_token_list.append(x)

        return aggregated_token_list
            
class PhysGM(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.num_global_tokens = config.model.num_global_tokens
        self.use_dinov3 = config.model.get("use_dinov3", False)
        self.post_act = config.model.get("post_act", False)

        if isinstance(config.model.dim, int):
            self.dim_start = config.model.dim
            self.dim_out = config.model.dim
        else:
            self.dim_start = config.model.dim[0]
            self.dim_out = config.model.dim[-1]

        dinov3_path = config.model.get("dinov3_path", "facebook/dinov3-base")
        dinov3_dim = config.model.get("dinov3_dim", 768)  # DINOv3-base: 768, large: 1024
        self.dinov3_extractor = DINOv3FeatureExtractor(dinov3_path, device)
        self.ray_encoder = DenseRepresentationEncoder(
            in_chans=6,
            embed_dim=32,
            patch_size=16,
            intermediate_dims=[128],
            apply_pe=True,
            input_size_for_pe=256
        )

        self.dino_proj = nn.Linear(dinov3_dim + 32, self.dim_start, bias=False) # dinov3 + plucker ray embed -> 1024
        self.input_layernorm = nn.LayerNorm(self.dim_start, bias=False)
        self.dpt_head = DPTHead(dim_in=self.dim_out, output_dim=(3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1))

        if self.num_global_tokens > 0:
            self.global_token_init = nn.Parameter(torch.randn(1, self.num_global_tokens, self.dim_start)*0.02)
            nn.init.constant_(self.global_token_init, 0.0)

        self.processor = Processor(config)
        if self.post_act:
            self.pos_act = lambda x: x.clamp(-1, 1)
            self.scale_act = lambda x: 0.1 * F.softplus(x)
            self.opacity_act = lambda x: torch.sigmoid(x)
            self.rot_act = lambda x: F.normalize(x, dim=-1)
            self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 

        self.phys_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim_out, bias=False),
            nn.Linear(
                self.dim_out, 14,
                bias=False,
            )
        )

        self.E_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim_out, bias=False),
            nn.Linear(self.dim_out, 64), 
            nn.GELU(),
            nn.Linear(64, 2, bias=True) 
        )
        
        self.nu_token_decoder = nn.Sequential(
            nn.LayerNorm(self.dim_out, bias=False),
            nn.Linear(self.dim_out, 64),
            nn.GELU(),
            nn.Linear(64, 2, bias=True)
        )
        
        self.phys_token_decoder.apply(_init_weights)
        nn.init.normal_(self.E_token_decoder[-1].weight, std=0.01) 
        nn.init.constant_(self.E_token_decoder[-1].bias, 0.0)      
        
        nn.init.normal_(self.nu_token_decoder[-1].weight, std=0.01)
        nn.init.constant_(self.nu_token_decoder[-1].bias, 0.0)

        if config.training.get("lpips_loss", 0.0) > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.eval()
            self.lpips_loss.requires_grad_(False)
    
    
            

    def robust_gaussian_nll_loss(self, mu, var, target):
        mu = mu.float()
        var = var.float()
        target = target.float()
        const = 0.5 * math.log(2 * math.pi)
        eps = 1e-2 
        var = torch.clamp(var, min=eps) 
        nll = const + 0.5 * torch.log(var) + 0.5 * (target - mu)**2 / (var)
        nll = torch.clamp(nll,max=100.0)
        return nll.mean()


    def forward(self, input_dict):
        """
        input_images: (B, V, 3, H, W)
        input_intr: (B, V, 4), (fx, fy, cx, cy)
        input_c2ws: (B, V, 4, 4)
        pos_avg_inv: (B, 4, 4)
        scene_scale: (B)
        """
        input_dict = edict(input_dict)
        input_images = input_dict["input_images"]
        input_intr = input_dict["input_intr"]
        input_c2ws = input_dict["input_c2ws"]
        test_images = input_dict.get("test_images", None)
        test_intr = input_dict.get("test_intr", None)
        test_c2ws = input_dict.get("test_c2ws", None)
        pos_avg_inv = input_dict.get("pos_avg_inv", None)
        scene_scale = input_dict.get("scene_scale", None)
        scene_name = input_dict.get("scene_name", None)
        use_checkpoint = input_dict.get("use_checkpoint", False)
        gt_phys = input_dict.get("physical_class", None)
        E_gt = input_dict.get("E", None)
        nu_gt = input_dict.get("nu", None)

        inference_start = time.time()
        B, V, C, H, W = input_images.shape

        if C == 4:
            bg_color = torch.ones(3, dtype=input_images.dtype, device=input_images.device)
            bg_color_view = bg_color.view(1, 1, 3, 1, 1)
            input_rgb = input_images[:, :, :3, :, :]    # [B, V, 3, H, W]
            input_alpha = input_images[:, :, 3:, :, :]  # [B, V, 1, H, W]
            input_images = input_rgb * input_alpha + bg_color_view * (1.0 - input_alpha)
            
        input_images_flat = input_images.reshape(B*V, 3, H, W) # (B*V,3,H,W)
        dinov3_features = self.dinov3_extractor(input_images_flat)  # (B*V, H//16, W//16, dinov3_dim)
        dinov3_flat = rearrange(dinov3_features, '(b v) h w c -> b v (h w) c', b=B, v=V)

        # Embed camera info
        ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float() # (B, V, H*W, 3) # camera origin
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = (x.to(input_intr.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device).contiguous()
        y = (y.to(input_intr.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device).contiguous()
        # unproject to camera space
        x = (x - input_intr[:, :, 2:3]) / input_intr[:, :, 0:1]
        y = (y - input_intr[:, :, 3:4]) / input_intr[:, :, 1:2]
        ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (B, V, H*W, 3)
        ray_d = F.normalize(ray_d, p=2, dim=-1)
        ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2).contiguous() # (B, V, H*W, 3)

        # test scene scale
        cam_points = test_c2ws[:, :, :3, 3]
        scene_scale = (cam_points[:, None, :, :] - cam_points[:, :, None, :]).norm(dim=-1).max().item()

        pluckey_tensor = torch.cat([torch.cross(ray_o, ray_d, dim=-1), ray_d], dim=-1) # (B, V, H*W, 6)
        pluckey_ray = rearrange(pluckey_tensor, 'b v (h w) c -> (b v) c h w', h=H, w=W)
        encoder_output = self.ray_encoder(pluckey_ray) # (B*V, C_out, H, W)
        ray_embed = rearrange(encoder_output, '(b v) c h w -> b v (h w) c', b=B, v=V)

        input_image_cam = torch.cat([
            dinov3_flat,  # dinov3 embed
            ray_embed,  # plucker ray embed
        ], dim=-1)  # (B, V, h*w, dinov3_dim+ray_dim)
        image_tokens = input_image_cam.flatten(1, 2)
        image_tokens = self.dino_proj(image_tokens)
            
        if self.num_global_tokens > 0:
            global_tokens = self.global_token_init.expand(B, -1, -1)
            tokens = torch.cat([global_tokens, image_tokens], dim=1) # (B, num_global_tokens+V*h*w, D)
        else:
            tokens = image_tokens

        tokens = self.input_layernorm(tokens)

        # Process tokens
        aggregated_tokens = self.processor(tokens)

        # Decode tokens
        phys_tokens = aggregated_tokens[-1][:, 0]
        E_tokens = aggregated_tokens[-1][:, 1]
        nu_tokens = aggregated_tokens[-1][:, 2]
        image_tokens = [_tokens[:, self.num_global_tokens:].reshape(B, V, -1, self.dim_out) for _tokens in aggregated_tokens] # (B, V*hh*ww, D)

        gaussians = self.dpt_head(image_tokens, input_images)
        gaussians = rearrange(gaussians, "b v h w d -> b (v h w) d")

        D_sh = (self.config.model.gaussians.sh_degree + 1) ** 2
        xyz, feature, scale, rotation, opacity = torch.split(
            gaussians,
            [3, D_sh * 3, 3, 4, 1],
            dim=-1
        )
        feature = feature.view(B, V*H*W, D_sh, 3).contiguous()

        phys_logits = self.phys_token_decoder(phys_tokens)
        E_logits = self.E_token_decoder(E_tokens)
        E_mu = E_logits[... , 0]
        E_var = F.softplus(E_logits[..., 1]) + 1e-2 
        nu_logits = self.nu_token_decoder(nu_tokens)
        nu_mu = nu_logits[... , 0]
        nu_var = F.softplus(nu_logits[..., 1]) + 1e-2

        if self.post_act:
            xyz = self.pos_act(xyz)
            opacity = self.opacity_act(opacity)
            scale = self.scale_act(scale)
            rotation = self.rot_act(rotation)
            feature = self.rgb_act(feature)

        img_rgb = rearrange(input_images, "b v c h w -> b (v h w) c")  # B, V*H*W, 3
        feature[:, :, 0, :] = feature[:, :, 0, :] + img_rgb

        scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max) 
        opacity = opacity + self.config.model.gaussians.opacity_bias
        
        # Align gaussian means to pixel centers
        if self.config.model.gaussians.get("align_to_pixel", True):
            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.config.model.gaussians.max_dist # (B, V*H*W, 1)
            xyz = dist * ray_d.reshape(B, -1, 3) + ray_o.reshape(B, -1, 3) # (B, V*H*W, 3)

        gaussians = {
            "xyz": xyz.float(),
            "feature": feature.float(),
            "scale": scale.float(),
            "rotation": rotation.float(),
            "opacity": opacity.float()
        }
        inference_time = time.time() - inference_start

        # GS Pruning
        num_gaussians = xyz.shape[1]
        prune_ratio = self.config.model.gaussians.get("prune_ratio", 0.0)
        gaussian_usage = (opacity.sigmoid() > self.config.model.gaussians.get("opacity_threshold", 0.001)).float().mean(dim=1).squeeze(-1) # (B,)
        if prune_ratio > 0:
            keep_ratio = 1 - prune_ratio
            random_ratio = self.config.model.gaussians.get("random_ratio", 0.0)
            random_ratio = keep_ratio * random_ratio
            keep_ratio = keep_ratio - random_ratio
            num_keep = int(num_gaussians * keep_ratio)
            num_keep_random = int(num_gaussians * random_ratio)
            idx_sort = opacity.argsort(dim=1, descending=True)
            keep_idx = idx_sort[:, :num_keep]
            if num_keep_random > 0:
                rest_idx = idx_sort[:, num_keep:]
                random_idx = rest_idx[:, torch.randperm(rest_idx.shape[1])[:num_keep_random]]
                keep_idx = torch.cat([keep_idx, random_idx], dim=1)
            for k, v in gaussians.items():
                v_shape = v.shape
                v = v.reshape(v_shape[0], v_shape[1], -1)
                v = v.gather(1, keep_idx.expand(-1, -1, v.shape[-1]))
                gaussians[k] = v.reshape(v_shape[0], -1, *v_shape[2:])

        ret_dict = {
            "gaussians":gaussians,
            "gaussian_usage": gaussian_usage,
            "inference_time": inference_time
        }

        ret_dict["E_mu"] = E_mu
        ret_dict["E_var"] = E_var
        ret_dict["nu_mu"] = nu_mu
        ret_dict["nu_var"] = nu_var
        if 'phys_logits' in locals(): 
            ret_dict["phys_logits"] = phys_logits
            
        if pos_avg_inv is not None:
            ret_dict["pos_avg_inv"] = pos_avg_inv

        if scene_scale is not None:
            ret_dict["scene_scale"] = scene_scale

        if scene_name is not None:
            ret_dict["scene_name"] = scene_name

        if test_c2ws is None:
            return ret_dict

        # Render images at test views
        xyz = gaussians["xyz"]
        feature = gaussians["feature"]
        scale = gaussians["scale"]
        rotation = gaussians["rotation"]
        opacity = gaussians["opacity"]

        def render_single_item(xyz_i, feature_i, scale_i, rotation_i, opacity_i, test_c2ws_i, test_intr_i):

            num_views = test_c2ws_i.shape[0]
            test_w2c_i = test_c2ws_i.float().inverse()
            test_intr_i_mat = torch.zeros(num_views, 3, 3, device=xyz_i.device, dtype=torch.float32)
            test_intr_i_mat[:, 0, 0] = test_intr_i[:, 0]
            test_intr_i_mat[:, 1, 1] = test_intr_i[:, 1]
            test_intr_i_mat[:, 0, 2] = test_intr_i[:, 2]
            test_intr_i_mat[:, 1, 2] = test_intr_i[:, 3]
            test_intr_i_mat[:, 2, 2] = 1
            
            # gsplat.rasterization 
            rendering, render_alpha, _ = rasterization(
                means=xyz_i,
                quats=F.normalize(rotation_i, p=2, dim=-1),
                scales=scale_i.exp(),
                opacities=opacity_i.sigmoid().squeeze(-1),
                colors=feature_i,
                viewmats=test_w2c_i,
                Ks=test_intr_i_mat,
                width=W,
                height=H,
                sh_degree=self.config.model.gaussians.sh_degree,
                near_plane=self.config.model.gaussians.near_plane,
                far_plane=self.config.model.gaussians.far_plane,
                render_mode="RGB",
                backgrounds=torch.ones(num_views, 3, device=xyz_i.device, dtype=torch.float32)
            )
            return torch.cat([rendering, render_alpha], dim=-1)

        renderings_list = []
        with torch.autocast(enabled=False, device_type="cuda"):
            for i in range(B):
                if use_checkpoint and self.training:
                    rendering_rgba = torch.utils.checkpoint.checkpoint(
                        render_single_item,
                        xyz[i], feature[i], scale[i], rotation[i], opacity[i], test_c2ws[i], test_intr[i],
                        use_reentrant=False
                    )
                else:
                    rendering_rgba = render_single_item(
                        xyz[i], feature[i], scale[i], rotation[i], opacity[i], test_c2ws[i], test_intr[i]
                    )
                renderings_list.append(rendering_rgba)

        renderings = torch.stack(renderings_list, dim=0) # (B, V, H, W, 4)


        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous() # (B, V, 4, H, W)
        ret_dict["renderings"] = renderings

        if test_images is None:
            return ret_dict

        if not self.training:
            return ret_dict

        # Compute loss
        bg_color = torch.ones(3, dtype=renderings.dtype, device=renderings.device)
        bg_color_view = bg_color.view(1, 1, 3, 1, 1)
        pred_rgb = renderings[:, :, :3, :, :]  # [B, V, 3, H, W]
        pred_alpha = renderings[:, :, 3:, :, :] # [B, V, 1, H, W]
        gt_rgb = test_images[:, :, :3, :, :]    # [B, V, 3, H, W]
        gt_alpha = test_images[:, :, 3:, :, :]  # [B, V, 1, H, W]
        gt_composited = gt_rgb * gt_alpha + bg_color_view * (1.0 - gt_alpha)

        l2_loss_rgb = F.mse_loss(pred_rgb, gt_composited)
        with torch.no_grad():
            psnr = -10 * torch.log10(l2_loss_rgb)

        total_loss = l2_loss_rgb * self.config.training.get("l2_loss", 1.0)

        loss_dict = {
            "l2_loss": l2_loss_rgb,
            "psnr": psnr,
        }

        renderings = renderings.flatten(0, 1) # (B*V, 4, H, W)
        test_images = gt_composited.flatten(0, 1) # (B*V, 4, H, W) #change

        if self.config.training.get("lpips_loss", 0.0) > 0 :
            lpips_renderings = torch.clamp(renderings[:, :3] * 2 - 1, min=-1.0, max=1.0)
            lpips_test = torch.clamp(test_images[:, :3] * 2 - 1, min=-1.0, max=1.0)
            
            loss_lpips = self.lpips_loss(
                lpips_renderings,
                lpips_test,
            ).mean()
            total_loss += loss_lpips * self.config.training.lpips_loss
            loss_dict["loss_lpips"] = loss_lpips

        if self.config.training.get("opacity_loss", 0.0) > 0:
            opacity_loss = F.mse_loss(pred_alpha, gt_alpha)
            total_loss += opacity_loss * self.config.training.opacity_loss
            loss_dict["opacity_loss"] = opacity_loss

        if self.config.training.get("physical_loss", 0.0) > 0:
            gt_phys = F.one_hot(gt_phys, num_classes=14)
            phys_loss = F.binary_cross_entropy_with_logits(phys_logits, gt_phys.float(), reduction='mean')
            total_loss += phys_loss * self.config.training.physical_loss
            loss_dict["phys_loss"] = phys_loss

        if self.config.training.get("E_loss", 0.0) > 0:
            
            # NLL Loss
            E_nll = self.robust_gaussian_nll_loss(E_mu, E_var, E_gt)
            loss_dict["E_nll"] = E_nll
            
            # MSE Loss 
            E_mse = F.mse_loss(E_mu.float(), E_gt.float())
            loss_dict["E_mse"] = E_mse
            
            
            mse_weight = 1.0 
            E_total_loss = E_nll + mse_weight * E_mse
            
            total_loss += E_total_loss * self.config.training.E_loss
            loss_dict["E_loss"] = E_total_loss 
        
        if self.config.training.get("nu_loss", 0.0) > 0:
            
            # NLL Loss
            nu_nll = self.robust_gaussian_nll_loss(nu_mu, nu_var, nu_gt)
            loss_dict["nu_nll"] = nu_nll
            
            # MSE Loss 
            nu_mse = F.mse_loss(nu_mu.float(), nu_gt.float())
            loss_dict["nu_mse"] = nu_mse

            mse_weight = 1.0 
            nu_total_loss = nu_nll + mse_weight * nu_mse

            
            total_loss += nu_total_loss * self.config.training.nu_loss
            loss_dict["nu_loss"] = nu_total_loss 
            
        
        loss_dict["total_loss"] = total_loss
        ret_dict["loss"] = loss_dict

        return ret_dict

    def save_gaussian_ply(self, gaussian_dict, save_path, opacity_threshold=None):
        """
        Adapted from the original 3D GS implementation
        https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
        """
        from plyfile import PlyData, PlyElement
        xyz = gaussian_dict["xyz"].detach().cpu().float() # (N, 3)
        normal = torch.zeros_like(xyz) # (N, 3)
        N = xyz.shape[0]
        feature = gaussian_dict["feature"].detach().cpu().float() # (N, (sh_degree+1)**2, 3)
        f_dc = feature[:, 0].contiguous() # (N, 3)
        f_rest_full = torch.zeros(N, 3*(3+1)**2-3).float()
        if feature.shape[1] > 1:
            f_rest = feature[:, 1:].transpose(1, 2).reshape(N, -1) # (N, 3*(sh_degree+1)**2-3)
            f_rest_full[:, :f_rest.shape[1]] = f_rest
        f_rest_full = f_rest_full.contiguous()
        scale = gaussian_dict["scale"].detach().cpu().float() # (N, 3)
        opacity = gaussian_dict["opacity"].detach().cpu().float() # (N, 1)
        rotation = gaussian_dict["rotation"].detach().cpu().float() # (N, 4)
        attributes = np.concatenate([xyz.numpy(), 
                                     normal.numpy().astype(np.uint8),
                                     f_dc.numpy(),
                                     f_rest_full.numpy(),
                                     opacity.numpy(),
                                     scale.numpy(),
                                     rotation.numpy()
                                    ], axis=1)
        if opacity_threshold is not None:                             
            attributes = attributes[opacity.squeeze(-1).sigmoid().numpy() > opacity_threshold]
        attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attribute_list += ['f_dc_{}'.format(i) for i in range(f_dc.shape[1])]
        attribute_list += ['f_rest_{}'.format(i) for i in range(f_rest_full.shape[1])]
        attribute_list += ['opacity']
        attribute_list += ['scale_{}'.format(i) for i in range(scale.shape[1])]
        attribute_list += ['rot_{}'.format(i) for i in range(rotation.shape[1])]
        dtype_full = [(attribute, 'f4') for attribute in attribute_list]
        dtype_full[3:6] = [(attribute, 'u1') for attribute in attribute_list[3:6]]
        elements = np.empty(attributes.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)


    def save_visualization(self, input_dict, output_dict, save_dir, save_gaussian=False, save_video=False):
        import torchvision
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        input_images = input_dict["input_images"] # (B, V, 4, H, W)
        target_images = input_dict["test_images"] # (B, V, 4, H, W)
        renderings = output_dict["renderings"] # (B, V, 4, H, W)
        
        B, V, _, H, W = target_images.shape

        # save images of first batch
        input_image_path = os.path.join(save_dir, "input_images.png")
        input_image = input_images[0][:, :4, :, :].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, Vin*W)
        torchvision.utils.save_image(input_image, input_image_path)
        target_rendering_path = os.path.join(save_dir, "target_renderings.png")
        target_renderings = []
        bg_color = torch.ones(3, dtype=renderings.dtype, device=renderings.device)
        bg_color_view = bg_color.view(3, 1, 1)
        for i in range(B):
            target_image_1 = target_images[i][:, :3, :, :].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
            target_alpha = target_images[i][:, 3:4, :, :].permute(1, 2, 0, 3).flatten(2, 3) # (1, H, V*W)
            target_image = target_image_1 * target_alpha + bg_color_view * (1.0 - target_alpha)
            rendering_image = renderings[i][:, :3, :, :].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
            target_renderings.append(target_image)
            target_renderings.append(rendering_image)
        target_rendering = torch.cat(target_renderings, dim=1) # (3, 2*B*H, V*W)
        
        torchvision.utils.save_image(target_rendering, target_rendering_path)
        

        # save gaussian ply of first batch
        if save_gaussian:
            gaussians = output_dict["gaussians"]
            camera_intrinsics = input_dict["input_intr"][0]
            camera_poses = input_dict["input_c2ws"][0]
            gaussian_first = {k: v[0] for k, v in gaussians.items()}
            opacity_threshold = self.config.model.gaussians.get("opacity_threshold", 0.001)
            self.save_gaussian_ply(gaussian_first, os.path.join(save_dir, f"gaussians_{str(opacity_threshold).split('.')[-1]}.ply"), opacity_threshold)

   


