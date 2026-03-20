import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights,vgg16


# evaluation metrics
from skimage.metrics import structural_similarity
from lpips import LPIPS

@torch.no_grad()
def compute_psnr(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((predict - target) ** 2, dim=(1, 2, 3)) # (B,)
    psnr = -10 * torch.log10(mse)
    return psnr

@torch.no_grad()
def compute_ssim(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    ssim = [
        structural_similarity(
            predict[i].cpu().numpy(),
            target[i].cpu().numpy(),
            multichannel=True,
            channel_axis=0,
            data_range=1.0,
        ) for i in range(predict.size(0))
    ]
    ssim = torch.tensor(ssim, device=predict.device, dtype=predict.dtype)
    return ssim

@torch.no_grad()
def compute_lpips(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    lpips_fn = LPIPS(net="vgg").to(predict.device)

    batch_size = 10
    values = []
    for i in range(0, predict.size(0), batch_size):
        value = lpips_fn.forward(
            predict[i : i + batch_size],
            target[i : i + batch_size],
            normalize=True,
        )
        values.append(value)
    value = torch.cat(values, dim=0)
    value = value[:, 0, 0, 0] # (B,)
    return value
        