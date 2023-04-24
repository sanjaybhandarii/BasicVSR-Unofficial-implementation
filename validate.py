import config
import torch
from torch import nn
from torch import Tensor
import numpy as np
from metrics import calc_psnr_ssim
from torchvision.utils import save_image
import wandb
import metrics
from tqdm import tqdm


val_losses = []
val_psnr = []
val_ssim = []
 
def validate(model,test_loader): 

    batch_idx = len(test_loader)        
    model.eval()

    ep_char_loss = 0        
    ep_psnr = 0
    ep_ssim = 0

    i=0
    with torch.no_grad():
        for batch, imgs in tqdm(enumerate(test_loader), total= batch_idx):
                high_curr = imgs['hr_input'].to(config.DEVICE)
                model_input = imgs['lr_input'].to(config.DEVICE)     
                char_loss = 0
                
                with torch.cuda.amp.autocast():
                    ssim = 0
                    psnr = 0
                    fake_curr = model(model_input)
                    seq_len = len(fake_curr)
                    for fake,high_img in zip(fake_curr, high_curr):
                        char_loss += config.charbonnier_loss(fake,high_img)
                        psnr_idx, ssim_idx = metrics.calc_psnr_ssim(fake, high_img)
                        ssim += ssim_idx
                        psnr += psnr_idx
                        
                char_loss = char_loss/seq_len
                ep_char_loss += char_loss.item()
                ep_psnr += (psnr/seq_len).item()
                ep_ssim += (ssim/seq_len).item()
                    


                for fake in fake_curr:
                    i += 1
                    j=0
                    for img in fake:
                        save_image(img, f"hr_{i}_{j}.png", normalize=False)
                        j += 1 
        

           
        ep_char_loss /= batch_idx
        ep_psnr /= batch_idx
        ep_ssim /= batch_idx
        
        val_losses.append(ep_char_loss)
        val_psnr.append(ep_psnr)
        val_ssim.append(ep_ssim)
        
        print(f'Val Char loss: {ep_char_loss} Val PSNR idx: {ep_psnr} Val SSIM idx: {ep_ssim}')
        wandb.log({'Val Char_loss':ep_char_loss,'Val PSNR_idx':ep_psnr,'Val ssim_idx':ep_ssim})
        np.savetxt("val_losses", val_losses, fmt='%f')
        np.savetxt("val_psnr", val_psnr, fmt='%f')
        np.savetxt("val_ssim", val_ssim, fmt='%f')
        return ep_char_loss,ep_psnr,ep_ssim


                
