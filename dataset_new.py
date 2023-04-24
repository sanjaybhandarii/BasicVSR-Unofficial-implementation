
import glob
import config

import random
from torch.autograd import Variable
import os
import numpy as np
from torchvision.utils import save_image, make_grid
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from image_preprocess import process_all_file
import threading
import queue

# Normalization parameters for pre-trained PyTorch models
# Normalization parameters for pre-trained PyTorch models
mean = [0.0, 0.0, 0.0]
std = [1.0, 1.0, 1.0]



class ImageDataset(Dataset):
    def __init__(self, hr_path, imgs_per_clip=15):
        '''
        Args:
        lr_path (str): Represents a path that contains a set of folders,
            where each folder contains a sequence of
            consecutive LR frames of a video.
        hr_path (str): See lr_path, but each folder
            contains HR frames. The folder and image names,
            when sorted, should be a 1:1 match with the LR frames
            (i.e. the third image in the second folder of the lr_path
            should be the LR image ofthe third image
            in the second folder of the hr_path).
        imgs_per_clip (int): The number of images that
            represents an input video. Default is 15,
            meaning each input video will consist of
            15 consecutive frames.
        '''
        self.hr_path = hr_path
        self.imgs_per_clip = imgs_per_clip
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.hr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.hr_folders) * self.clips_per_folder
        
        
        self.transform_hr = transforms.Compose(
        [
        #  transforms.Resize((config.HIGH_RES,config.HIGH_RES),interpolation=Image.BICUBIC),   
         transforms.ToTensor(),
        
        ]
        )
        
        self.transform_lr = transforms.Compose(
        [
         transforms.Resize((config.LOW_RES,config.LOW_RES),interpolation=Image.BICUBIC),   
         transforms.ToTensor(),
    
        
        ]
        )

    
    def __len__(self):
        return self.num_clips
    
    def augment(self,image,top,left,crop=True,hflip=True,vflip=True,transpose=True):
        if crop:
            # image = transforms.CenterCrop(256)(image)
            image = transforms.functional.crop(image, top=top, left=left, height=256, width=256)
        else:
            image = transforms.Resize((config.HIGH_RES,config.HIGH_RES),interpolation=Image.BICUBIC)(image)
        if hflip:
            image=transforms.RandomHorizontalFlip(p=1.0)(image)
        if vflip:
            image=transforms.RandomVerticalFlip(p=1.0)(image)
        
        if transpose:
            image = image.transpose(method=Image.Transpose.ROTATE_90)
        
        return image
            
    
    
    def __getitem__(self, idx):
        '''
        Returns a np.array of an input video that is of shape
        (T, H, W, 3), where T = imgs per clip, H/W = height/width,
        and 3 = channels (3 for RGB images). Note that the video
        pixel values will be between [0, 255], not [0, 1).
        '''
        lr_input=[]
        hr_input = []
        folder_idx, clip_idx = idx // self.clips_per_folder, idx % self.clips_per_folder
        s_i, e_i = self.imgs_per_clip * clip_idx, self.imgs_per_clip * (clip_idx + 1)
        # lr_fnames = sorted(glob.glob(f'{self.lr_folders[folder_idx]}/*'))[s_i:e_i]
        hr_fnames = sorted(glob.glob(f'{self.hr_folders[folder_idx]}/*'))[s_i:e_i]
        
        img = Image.open(hr_fnames[0])
        w,h= img.size
        top = random.randint(0,h-260)
        left = random.randint(0,w-260)
        
        crop = False
        rotate = False
        hflip = False
        vflip = False
        transpose = False
        
        
        if np.random.uniform()<0.5:
            crop=True
        if np.random.uniform()<0.5:
            hflip = True
        if np.random.uniform()<0.5:
            vflip = True
        if np.random.uniform()<0.5:
            transpose = True
        
        
        for i, hr_fname in enumerate(hr_fnames):
                    
            img = Image.open(hr_fname)
            img = self.augment(img,top=top,left = left,crop= crop,hflip=hflip,vflip= vflip,transpose=transpose)
            hr_img = self.transform_hr(img)
            hr_input.append(hr_img)
            
            img = self.transform_lr(img)
            lr_input.append(img)
            
        if np.random.uniform()<0.5:
            hr_input.reverse()
            lr_input.reverse()
        
               
        lr_input = torch.stack(lr_input,dim=0)
        hr_input = torch.stack(hr_input,dim=0) 
        return {"lr_input":lr_input, "hr_input": hr_input}
    
    

class ValImageDataset(Dataset):
    def __init__(self, hr_path, imgs_per_clip=15):
        '''
        Args:
        lr_path (str): Represents a path that contains a set of folders,
            where each folder contains a sequence of
            consecutive LR frames of a video.
        hr_path (str): See lr_path, but each folder
            contains HR frames. The folder and image names,
            when sorted, should be a 1:1 match with the LR frames
            (i.e. the third image in the second folder of the lr_path
            should be the LR image ofthe third image
            in the second folder of the hr_path).
        imgs_per_clip (int): The number of images that
            represents an input video. Default is 15,
            meaning each input video will consist of
            15 consecutive frames.
        '''
        self.hr_path = hr_path
        self.imgs_per_clip = imgs_per_clip
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.hr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.hr_folders) * self.clips_per_folder
        
        self.crop_hr = transforms.Compose(
        [
            transforms.CenterCrop(256),
        ]
        )
        
        self.transform_hr = transforms.Compose(
        [
         transforms.Resize((config.HIGH_RES,config.HIGH_RES),interpolation=Image.BICUBIC),   
         transforms.ToTensor(),
        
        ]
        )
        
        self.transform_lr = transforms.Compose(
        [
         transforms.Resize((config.LOW_RES,config.LOW_RES),interpolation=Image.BICUBIC),   
         transforms.ToTensor(),
    
        
        ]
        )

    
    def __len__(self):
        return self.num_clips
    
    
    
    
    def __getitem__(self, idx):
        '''
        Returns a np.array of an input video that is of shape
        (T, H, W, 3), where T = imgs per clip, H/W = height/width,
        and 3 = channels (3 for RGB images). Note that the video
        pixel values will be between [0, 255], not [0, 1).
        '''
        lr_input=[]
        hr_input = []
        folder_idx, clip_idx = idx // self.clips_per_folder, idx % self.clips_per_folder
        s_i, e_i = self.imgs_per_clip * clip_idx, self.imgs_per_clip * (clip_idx + 1)
        hr_fnames = sorted(glob.glob(f'{self.hr_folders[folder_idx]}/*'))[s_i:e_i]
       
     
    
        for i, hr_fname in enumerate(hr_fnames):
                    
            img = Image.open(hr_fname)
            # img = self.crop_hr(img)
            hr_img = self.transform_hr(img)
            hr_input.append(hr_img)
            
            img = self.transform_lr(img)
            lr_input.append(img)
                    
        lr_input = torch.stack(lr_input,dim=0)
        hr_input = torch.stack(hr_input,dim=0) 
        return {"lr_input":lr_input, "hr_input": hr_input}

class TestImageDataset(Dataset):
    def __init__(self, hr_path, imgs_per_clip=100):
        '''
        Args:
        lr_path (str): Represents a path that contains a set of folders,
            where each folder contains a sequence of
            consecutive LR frames of a video.
        hr_path (str): See lr_path, but each folder
            contains HR frames. The folder and image names,
            when sorted, should be a 1:1 match with the LR frames
            (i.e. the third image in the second folder of the lr_path
            should be the LR image ofthe third image
            in the second folder of the hr_path).
        imgs_per_clip (int): The number of images that
            represents an input video. Default is 15,
            meaning each input video will consist of
            15 consecutive frames.
        '''
        self.hr_path = hr_path
        self.imgs_per_clip = imgs_per_clip
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.hr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.hr_folders) * self.clips_per_folder
        
      
        
        self.transform_hr = transforms.Compose(
        [
         transforms.Resize((512,512),interpolation=Image.BICUBIC),   
         transforms.ToTensor(),
        
        ]
        )
        
        self.transform_lr = transforms.Compose(
        [
         transforms.Resize((128,128),interpolation=Image.BICUBIC),   
         transforms.ToTensor(),
    
        
        ]
        )

    
    def __len__(self):
        return self.num_clips
    
    
    
    
    def __getitem__(self, idx):
        '''
        Returns a np.array of an input video that is of shape
        (T, H, W, 3), where T = imgs per clip, H/W = height/width,
        and 3 = channels (3 for RGB images). Note that the video
        pixel values will be between [0, 255], not [0, 1).
        '''
        lr_input=[]
        hr_input = []
        folder_idx, clip_idx = idx // self.clips_per_folder, idx % self.clips_per_folder
        s_i, e_i = self.imgs_per_clip * clip_idx, self.imgs_per_clip * (clip_idx + 1)
        hr_fnames = sorted(glob.glob(f'{self.hr_folders[folder_idx]}/*'))[s_i:e_i]
       
     
    
        for i, hr_fname in enumerate(hr_fnames):
                    
            img = Image.open(hr_fname)
            # img = self.crop_hr(img)
            hr_img = self.transform_hr(img)
            hr_input.append(hr_img)
            
            img = self.transform_lr(img)
            lr_input.append(img)
                    
        lr_input = torch.stack(lr_input,dim=0)
        hr_input = torch.stack(hr_input,dim=0) 
        return {"lr_input":lr_input, "hr_input": hr_input}



class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.
    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self



class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler,
            and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
