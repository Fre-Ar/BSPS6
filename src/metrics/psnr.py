import torch
from torch.nn.functional import mse_loss as mse

@torch.no_grad()
def psnr(image_pred, image_gt):
    return -10*torch.log10(mse(image_pred, image_gt))