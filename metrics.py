import torch

from basicsr.metrics.niqe import calculate_niqe
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

#calculate avg of a list
avg = lambda x: sum(x)/len(x)


# def calc_niqe(img_list_tensor):
#     img_list_tensor = torch.nn.functional.relu(img_list_tensor)
#     print(img_list_tensor)
#     niqe = [calculate_niqe(255*img.cpu().numpy(),crop_border=0, input_order='CHW') for img in img_list_tensor]
    
#     return avg(niqe)

def calc_psnr_ssim(img_list_gt, img_list_hr):
    ssim_index =  calculate_ssim_pt(img_list_gt, img_list_hr,crop_border=0,test_y_channel=True)
    
    psnr_index= calculate_psnr_pt(img_list_gt, img_list_hr, crop_border=0,test_y_channel=True)
    
    return avg(psnr_index),avg(ssim_index)


