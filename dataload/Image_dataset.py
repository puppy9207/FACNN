import torch
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
import numpy as np
import os
from dataload.degradation import degradation_pipeline

def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

def degradation(image):
    image = Image.fromarray(degradation_pipeline(np.array(image)).astype(np.uint8))
    return image

class TrainDataset(torch.utils.data.dataset.Dataset):
    """ 학습 데이터 셋 """
    def __init__(self, opt, name,device):
        super(TrainDataset, self).__init__()
        # 파일 목록 작성
        self.filenames = [os.path.join(opt["train_root_path"], name, x) for x in os.listdir(os.path.join(opt["train_root_path"], name)) if check_image_file(x)]
        # 전처리 기법 등록
        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((opt["crop_size"] // opt["scale_factor"], opt["crop_size"] // opt["scale_factor"]), interpolation=Image.BICUBIC),
            transforms.Lambda(degradation),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((opt["crop_size"], opt["crop_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert("RGB")

        hr = self.hr_transforms(image)
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)
