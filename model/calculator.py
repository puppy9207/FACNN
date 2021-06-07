import numpy as np
from math import log10, sqrt

from numpy.core.fromnumeric import compress
from skimage.metrics import structural_similarity as cal_ssim
import lpips, torch

# PSNR 측정
def PSNR(original, compressed):
    """ PSNR 측정 함수 """
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

# SSIM 측정
def SSIM(original, compressed):
    """ SSIM 측정 함수 """
    ssim, diff = cal_ssim(original, compressed, multichannel=True, full=True)
    return ssim

def LPIPS(original, compressed, lpips_metric, device):
    """ LPIPS 측정 함수 """
    original = original.transpose(2, 0, 1)
    compressed = compressed.transpose(2, 0, 1)
    original_t = torch.from_numpy(original)
    compressed_t = torch.from_numpy(compressed)    
    lpips_value = lpips_metric(original_t.to(device), compressed_t.to(device))
    return lpips_value.cpu().detach().numpy()

def PSNR_tensor(original, compressed):
    return 10. * torch.log10(1. / torch.mean((original - compressed) ** 2))