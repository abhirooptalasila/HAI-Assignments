import os
import torch
import numpy as np 
from PIL import Image
from .preprocessing import *
from torch.utils.data.dataset import Dataset


class ISBI_D(Dataset):
    def __init__(self, pardir, transform=None):
        imgs = os.path.join(os.path.abspath(pardir), "images")
        self.imgs = [os.path.join(imgs, x)
         for x in os.listdir(imgs)]
        masks = os.path.join(os.path.abspath(pardir), "masks")
        self.masks = [os.path.join(masks, x)
         for x in os.listdir(masks)]
        self.data_len = len(self.imgs)
        self.transforms = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = Image.open(img_file)
        img_arr = np.asarray(img)

        mask_file = self.masks[index]
        mask = Image.open(mask_file)
        mask_arr = np.asarray(mask)

        if self.transforms:
            aug = self.transforms(image=img_arr, mask=mask_arr)
        return aug["image"], aug["mask"]


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "data", "isbi", "train")
    dd = ISBI_D(path)