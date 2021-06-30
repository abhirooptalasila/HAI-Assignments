import os
import torch
import numpy as np 
from .utils import *
from tqdm import tqdm
from PIL import Image
from skimage import io
from pathlib import Path
from torch.utils.data.dataset import Dataset


class ISBI(Dataset):
	def __init__(self, pardir, transform=None):
		imgs_path = os.path.join(os.path.abspath(pardir), "images")
		self.imgs = self.open_images(imgs_path)
		masks_path = os.path.join(os.path.abspath(pardir), "masks")
		self.masks = self.open_images(masks_path)

		# self.imgs = torch.stack([torch.Tensor(i) 
		# 	for i in imgs])
		# self.masks = torch.stack([torch.Tensor(i) 
		# 	for i in masks])
		self.data_len = len(self.imgs)
		self.transforms = transform

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		img = self.imgs[index]
		# img_arr = np.asarray(img)

		mask = self.masks[index]
		# mask_arr = np.asarray(mask)
    
		if self.transforms:
			aug = self.transforms(image=img, mask=mask)
			aug_img = torch.Tensor(aug["image"])
			aug_mask = torch.Tensor(aug["mask"])
			return torch.unsqueeze(aug_img, 0), torch.unsqueeze(aug_mask, 0)
			
		img = torch.Tensor(img)
		mask = torch.Tensor(mask)
		return torch.unsqueeze(img, 0), torch.unsqueeze(mask, 0)
	
	def open_images(self, path):
		x = np.array([np.array(Image.open(os.path.join(path, x)))/255 for x in os.listdir(path)])
		return x


class DS_Bowl(Dataset):
	def __init__(self, path, itransform=None, mtransform=None):
		self.data = self._load_data(path)
		self.itransform = itransform
		self.mtransform = mtransform

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		data = self.data[index]
		img = data['img'].numpy()
		masks = data['mask'].numpy()
    
		if self.itransform and self.mtransform:
			timg = self.itransform(image=img)
			mimg = self.mtransform(image=masks)
			itensor = torch.Tensor(timg["image"])
			itensor = itensor.reshape(
				itensor.shape[2], itensor.shape[0], itensor.shape[1])
			mtensor = torch.Tensor(mimg["image"])
			return itensor, torch.unsqueeze(mtensor, 0)

		imgt = torch.Tensor(img)
		imgt = imgt.reshape(imgt.shape[2], imgt.shape[0], imgt.shape[1])
		return imgt, torch.unsqueeze(torch.Tensor(masks), 0)
	
	def _load_data(self, file_path):
		file_path = Path(file_path)
		files = sorted(list(Path(file_path).iterdir()))
		files[:] = [file for file in files if not file.is_file()]
		datas = []

		for file in tqdm(files):
			item = {}
			imgs = []
			for image in (file/'images').iterdir():
				img = Image.open(image)
				imgs.append(np.array(img))
			img = np.array(img)
			img = img[:,:,:3]/255

			mask_files = list((file/'masks').iterdir())
			masks = None
			for ii,mask in enumerate(mask_files):
				mask = np.array(Image.open(mask).convert("L"))
				assert (mask[(mask!=0)]==255).all()
				if masks is None:
					H,W = mask.shape
					masks = np.zeros((len(mask_files), H, W))
				masks[ii] = mask
			for ii,mask in enumerate(masks):
				masks[ii] = mask/255 * (ii+1)
			mask = masks.sum(0)
			item['mask'] = torch.from_numpy(mask) #.squeeze(-1)
			item['mask'][item['mask'] > 0] = 1 # all same scale
			item['name'] = str(file).split('/')[-1]
			item['img'] = torch.from_numpy(img)
			datas.append(item)
		return datas


if __name__ == "__main__":
	path = os.path.join(os.getcwd(), "data", "isbi", "train")
	dd = ISBI(path)