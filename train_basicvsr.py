import wandb
from tqdm import tqdm
import torch
import numpy as np
import config
from validate import validate
from torch import Tensor
from torch import optim
from utils import load_model_opt, save_checkpoint
from torch.utils.data import DataLoader
import config
import metrics
from dataset_new import ValImageDataset,ImageDataset,TestImageDataset
from model_basicvsr import BasicVSRNet

wandb.init(project="New Basic VSR ", entity="basicvsr_unofficial")#define your wandb tracking
wandb.config = {
        "model_learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.TRAIN_BATCH_SIZE
        }

train_losses = []
train_ssim = []
train_psnr = []
best_losses = []
best_psnr = []
best_ssim = []


def train_fn(
    train_loader,
    val_loader,
    epochs,
    model,
    opt,
    scheduler,
    g_scaler 
):
    prev_val_loss,_,__ = validate(model,val_loader)
    prev_val_loss = 10
    batch_idx = len(train_loader)
    prev_train_loss = 10
    print("training started")
    for epoch in range(epochs):
        
        model.train()
        ep_char_loss = 0        
        ep_psnr = 0
        ep_ssim = 0
      
        for batch, imgs in tqdm(enumerate(train_loader), total= batch_idx):
          #  batch += 1
            if imgs['lr_input']==None:
                continue
            high_curr = Tensor(imgs['hr_input'].to(config.DEVICE))
            model_input = Tensor(imgs['lr_input'].to(config.DEVICE))         

            opt.zero_grad()
            with torch.cuda.amp.autocast():     
                ssim = 0
                psnr = 0
                char_loss = 0
                fake_curr = model(model_input)
                
                seq_len = len(fake_curr)
                
                for fake,high_img in zip(fake_curr, high_curr):
                    char_loss += config.charbonnier_loss(fake,high_img)
                    psnr_idx, ssim_idx = metrics.calc_psnr_ssim(fake, high_img)
                    ssim += ssim_idx
                    psnr += psnr_idx
                
                char_loss = char_loss/seq_len
                
                
                ep_char_loss += char_loss.item()
                g_scaler.scale(char_loss).backward()
                g_scaler.step(opt)
                g_scaler.update()
                
                scheduler.step()
                
               
                
                
                ep_psnr += (psnr/seq_len).item()
           
                ep_ssim += (ssim/seq_len).item()
            
        ep_char_loss /= batch_idx

        ep_psnr /= batch_idx
        ep_ssim /= batch_idx
        
        train_losses.append(ep_char_loss)
        train_psnr.append(ep_psnr)
        train_ssim.append(ep_ssim)
        
        print(f'Epoch: {epoch} Char loss: {ep_char_loss} PSNR idx: {ep_psnr} SSIM idx: {ep_ssim}')
        
        wandb.log({'Char_loss':ep_char_loss,'ssim_idx':ep_ssim,'psnr_idx':ep_psnr})
        
        np.savetxt("train_losses", train_losses, fmt='%f')
        np.savetxt("train_psnr", train_psnr, fmt='%f')
        np.savetxt("train_ssim", train_ssim, fmt='%f')
        
        if (ep_char_loss<prev_train_loss) or ((epoch+1)%5 == 0):
            prev_train_loss = ep_char_loss
            val_loss,val_psnr,val_ssim = validate(model,val_loader)
            
            
            if val_loss<prev_val_loss:
                best_losses.append(val_loss)
                best_psnr.append(val_psnr)
                best_ssim.append(val_ssim)
                
                save_checkpoint(model, opt, filename=config.CHECKPOINT_BEST_SAVE)
                prev_val_loss = val_loss
                np.savetxt("best_losses", best_losses, fmt='%f')
                np.savetxt("best_psnr", best_psnr, fmt='%f')
                np.savetxt("best_ssim", best_ssim, fmt='%f')
                
     
        
            
    return model, opt






def main():

 
    
    
    train_dataset =ImageDataset(config.TRAIN_HR_PATH)
    test_dataset = ValImageDataset(config.VAL_HR_PATH)


    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True,prefetch_factor=2,persistent_workers=True,num_workers=config.NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    

    #train_fetcher = CUDAPrefetcher(train_dataloader,device=config.DEVICE)

    model = BasicVSRNet()
    # model.spynet.requires_grad_(False)
    model = model.to(config.DEVICE)
    
    n_p = sum(p.numel() for p in model.parameters())
    print("Parameters : ",n_p)
    #opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt = optim.Adam(
    [
        {"params": model.spynet.parameters(), "lr": 2.5e-5},
        {"params": model.forward_resblocks.parameters()},
        {"params": model.backward_resblocks.parameters()},
        {"params": model.fusion.parameters()},
        {"params": model.upsample1.parameters()},
        {"params": model.upsample2.parameters()},
        {"params": model.conv_hr.parameters()},
        {"params": model.conv_last.parameters()},
        {"params": model.img_upsample.parameters()},
        {"params": model.lrelu.parameters()},
    ],
    lr=2e-4,
    betas=(0.9, 0.99)
    )
    
    
    
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40000, eta_min=1e-7, last_epoch=-1, verbose=False)
    
    g_scaler = torch.cuda.amp.GradScaler()
   

    #g_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[800000,], gamma=0.5)
   
    if config.LOAD_MODEL:
        load_model_opt(
            config.CHECKPOINT_BEST_LOAD,
            model,
            opt,
            config.LEARNING_RATE,
        )
        
        
    model, opt = train_fn(
            train_dataloader,
            test_dataloader,
            config.NUM_EPOCHS,
            model,
            opt,
            sched,
            g_scaler,
            )



if __name__ == "__main__":
    main()
