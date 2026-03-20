import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union

# --- 辅助模块：残差块 ---
class ResidualBlock(nn.Module):
    """
    一个简化的残差块。
    包含两个3x3卷积层和一个短路连接 (shortcut connection)。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 如果输入和输出通道数不同，则使用1x1卷积来匹配维度
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x + shortcut
        return self.act(x)

class DenseRepresentationEncoder(nn.Module):
    """
    一个简化的密集表示编码器 (Dense Representation Encoder)。

    该编码器将一个 (B, C, H, W) 的图像张量转换为一个密集特征图。
    核心功能包括：
    1. 使用 PixelUnshuffle 和卷积将图像转换为 patch 嵌入。
    2. 通过一系列残差块处理这些嵌入。
    3. （可选）添加可插值的正弦位置编码，以适应不同尺寸的输入。
    """
    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        patch_size: int = 16,
        intermediate_dims: List[int] = [128],
        apply_pe: bool = True,
        input_size_for_pe: Union[int, Tuple[int, int]] = 256,
    ):
        """
        构造函数。

        Args:
            in_chans (int): 输入图像的通道数。
            embed_dim (int): 最终输出特征的维度。
            patch_size (int): 每个 patch 的边长。
            intermediate_dims (List[int]): 卷积网络中间层的通道维度。
            apply_pe (bool): 是否应用位置编码。
            input_size_for_pe (int or Tuple[int, int]): 用于生成基础位置编码的图像尺寸。
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.apply_pe = apply_pe

        # 1. Patch 嵌入层
        # 使用 PixelUnshuffle 将 (B, C, H*P, W*P) -> (B, C*P*P, H, W)
        # 接着用一个卷积将通道数调整到第一个中间维度
        self.patch_embed = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(in_chans * patch_size**2, intermediate_dims[0], kernel_size=3, padding=1),
            nn.GELU()
        )

        # 2. 特征提取网络 (一系列残差块 + 最终的1x1卷积)
        layers = []
        # 添加残差块
        prev_dim = intermediate_dims[0]
        for dim in intermediate_dims[1:]:
            layers.append(ResidualBlock(prev_dim, dim))
            prev_dim = dim
        
        # 添加最终的1x1卷积层，将维度映射到 embed_dim
        layers.append(nn.Conv2d(prev_dim, embed_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

        # 3. 归一化层
        self.norm = nn.LayerNorm(embed_dim)

        # 4. 位置编码 (如果启用)
        if self.apply_pe:
            pe_input_size = (input_size_for_pe, input_size_for_pe) if isinstance(input_size_for_pe, int) else input_size_for_pe
            self.pe_patches_resolution = (pe_input_size[0] // patch_size, pe_input_size[1] // patch_size)
            num_patches = self.pe_patches_resolution[0] * self.pe_patches_resolution[1]
            
            # 注册一个 buffer 来保存位置编码，这样它会随模型移动 (e.g., to GPU)，但不是模型参数
            self.register_buffer("pos_embed", self._get_sinusoid_encoding_table(num_patches, embed_dim))
            self.post_pe_norm = nn.LayerNorm(embed_dim)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """生成正弦位置编码表"""
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数索引使用 sin
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数索引使用 cos
        return torch.FloatTensor(sinusoid_table)

    def _interpolate_pos_encoding(self, features, height, width):
        """
        将位置编码插值到当前特征图的尺寸。
        """
        previous_dtype = features.dtype
        npatch = features.shape[1]
        
        # 如果当前 patch 数量和预设的 PE patch 数量相同，则直接返回
        if npatch == self.pos_embed.shape[0]:
            return self.pos_embed.unsqueeze(0).to(previous_dtype)

        # 否则，进行双三次插值
        dim = features.shape[-1]
        # 计算当前特征图的 patch 分辨率
        h0, w0 = height // self.patch_size, width // self.patch_size
        
        # 获取预设的 PE patch 分辨率
        pe_h, pe_w = self.pe_patches_resolution
        
        # 将 PE 从 (N, C) 变形为 (1, C, H, W) 以便插值
        pos_embed = self.pos_embed.reshape(1, pe_h, pe_w, dim).permute(0, 3, 1, 2)
        
        # 插值
        pos_embed = nn.functional.interpolate(
            pos_embed,
            size=(h0, w0),
            mode='bicubic',
            antialias=True, # 使用 antialias 提高插值质量
        )
        
        # 将形状变回 (1, N, C)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出特征图，形状为 (B, embed_dim, H/patch_size, W/patch_size)。
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"输入图像尺寸 ({H}, {W}) 必须能被 patch_size ({self.patch_size}) 整除。"

        # 1. 编码为 patch 特征: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        features = self.patch_embed(x)
        features = self.encoder(features)

        # 2. 变形并归一化: (B, E, H/P, W/P) -> (B, N, E) where N = (H/P)*(W/P)
        # flatten(2) 将 H 和 W 维度展平，transpose(1, 2) 交换序列和嵌入维度
        features = features.flatten(2).transpose(1, 2)
        features = self.norm(features)

        # 3. （可选）应用位置编码
        if self.apply_pe:
            pos_embed = self._interpolate_pos_encoding(features, H, W)
            features = features + pos_embed
            features = self.post_pe_norm(features)

        # 4. 恢复形状: (B, N, E) -> (B, E, H/P, W/P)
        features = features.transpose(1, 2)
        features = features.view(B, self.embed_dim, H // self.patch_size, W // self.patch_size)

        return features

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 初始化一个不带位置编码的编码器
    encoder_no_pe = DenseRepresentationEncoder(
        in_chans=3,
        embed_dim=768,
        patch_size=14,
        intermediate_dims=[256, 512],
        apply_pe=False # 禁用 PE
    )
    
    # 2. 初始化一个带位置编码的编码器
    encoder_with_pe = DenseRepresentationEncoder(
        in_chans=3,
        embed_dim=1024,
        patch_size=14,
        intermediate_dims=[588, 768, 1024],
        apply_pe=True,
        input_size_for_pe=518 # 基础 PE 对应的图像大小
    )

    # 3. 测试不同尺寸的输入
    print("--- 测试带位置编码的编码器 ---")
    
    # 输入尺寸与 PE 基础尺寸相同
    dummy_image_1 = torch.randn(1, 3, 518, 518)
    output_1 = encoder_with_pe(dummy_image_1)
    print(f"输入尺寸: {dummy_image_1.shape}")
    print(f"输出特征图尺寸: {output_1.shape}\n") # 预期输出: (1, 1024, 37, 37)

    # 输入尺寸不同，将触发 PE 插值
    dummy_image_2 = torch.randn(1, 3, 224, 224)
    output_2 = encoder_with_pe(dummy_image_2)
    print(f"输入尺寸: {dummy_image_2.shape}")
    print(f"输出特征图尺寸: {output_2.shape}\n") # 预期输出: (1, 1024, 16, 16)
    
    print("模型初始化和测试成功!")