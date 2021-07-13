import os
import torch
import numpy as np 
from .utils import *
from tqdm import tqdm
from PIL import Image
from skimage import io
from pathlib import Path
from os.path import join, abspath
from torch.utils.data.dataset import Dataset


class ISBI(Dataset):
	def __init__(self, pardir, transform=None):
		imgs_path = join(abspath(pardir), "images")
		self.imgs = self.open_images(imgs_path)
		masks_path = join(abspath(pardir), "masks")
		self.masks = self.open_images(masks_path)

		self.data_len = len(self.imgs)
		self.transforms = transform

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		img = self.imgs[index]
		mask = self.masks[index]
    
		if self.transforms:
			aug = self.transforms(image=img, mask=mask)
			aug_img = torch.Tensor(aug["image"])
			aug_mask = torch.Tensor(aug["mask"])
			return torch.unsqueeze(aug_img, 0), torch.unsqueeze(aug_mask, 0)
			
		img = torch.Tensor(img)
		mask = torch.Tensor(mask)
		return torch.unsqueeze(img, 0), torch.unsqueeze(mask, 0)
	
	def open_images(self, path):
		x = np.array([np.array(Image.open(
				join(path, x)))/255 for x in os.listdir(path)])
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
			#itensor = itensor.reshape(
			#	itensor.shape[2], itensor.shape[0], itensor.shape[1])
			mtensor = torch.Tensor(mimg["image"])
			return torch.unsqueeze(itensor, 0), torch.unsqueeze(mtensor, 0)

		imgt = torch.Tensor(img)
		#imgt = imgt.reshape(imgt.shape[2], imgt.shape[0], imgt.shape[1])
		return torch.unsqueeze(imgt, 0), torch.unsqueeze(torch.Tensor(masks), 0)
	
	def _load_data(self, file_path):
		file_path = Path(file_path)
		files = sorted(list(Path(file_path).iterdir()))
		files[:] = [file for file in files if not file.is_file()]
		datas = []

		for file in tqdm(files):
			item = {}
			imgs = []
			for image in (file/'images').iterdir():
				img = Image.open(image).convert("L")
				imgs.append(np.array(img))
			img = np.array(img)
			#img = img[:,:,:3]/255
			img = img/255

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


class ResNetDS_Bowl(Dataset):
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
		return imgt, torch.unsqueeze(masks, 0)
	
	def _load_data(self, file_path):
		file_path = Path(file_path)
		files = sorted(list(Path(file_path).iterdir()))
		files[:] = [file for file in files if not file.is_file()]
		datas = []

		for file in tqdm(files):
			item = {}
			imgs = []
			for image in (file/'images').iterdir():
				img = io.imread(image) #.convert("L")
				imgs.append(img)
			#img = np.array(img)
			img = img[:,:,:3]/255
			
			mask_files = list((file/'masks').iterdir())
			masks = None
			for ii,mask in enumerate(mask_files):
				mask = np.array(Image.open(mask)) #.convert("L"))
				assert (mask[(mask!=0)]==255).all()
				if masks is None:
					H,W = mask.shape
					masks = np.zeros((len(mask_files), H, W))
				masks[ii] = mask
			for ii,mask in enumerate(masks):
				masks[ii] = mask/255 * (ii+1)
			mask = masks.sum(0)
			mask = mask / np.max(mask)
			item['mask'] = torch.from_numpy(mask) #.squeeze(-1)
			#item['mask'][item['mask'] > 0] = 1 # all same scale
			item['name'] = str(file).split('/')[-1]
			item['img'] = torch.from_numpy(img)
			datas.append(item)
		return datas


class LITS(Dataset):
	def __init__(self, pardir, transform=None):
		self.imgs = [join(pardir, "images", x) 
			for x in os.listdir(join(pardir, "images"))]
		self.masks = [join(pardir, "masks", x) 
			for x in os.listdir(join(pardir, "masks"))]

		self.data_len = len(self.imgs)
		self.transforms = transform

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		img = self.open_image(self.imgs[index])
		mask = self.open_image(self.masks[index])
		mask[mask < 0.3] = 0
		mask[mask >= 0.3] = 1

		if self.transforms:
			aug = self.transforms(image=img, mask=mask)
			aug_img = torch.Tensor(aug["image"])
			aug_mask = torch.Tensor(aug["mask"])
			return torch.unsqueeze(aug_img, 0), torch.unsqueeze(aug_mask, 0)
			
		img = torch.Tensor(img)
		mask = torch.Tensor(mask)
		return torch.unsqueeze(img, 0), torch.unsqueeze(mask, 0)
	
	def open_image(self, path):
		x = np.array(Image.open(path).convert("L"))/255
		return x


class BRATS(Dataset):
	def __init__(self, pardir, transform=None):
		self.imgs = np.load(join(pardir,  "X.npy"))
		self.masks = np.load(join(pardir, "Y.npy"))

		self.data_len = len(self.imgs)
		self.transforms = transform

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		img = self.imgs[index]/255
		mask = self.masks[index]/255
		mask[mask > 0.2] = 1; mask[mask <= 0.2] = 0

		if self.transforms:
			aug = self.transforms(image=img, mask=mask)
			aug_img = torch.Tensor(aug["image"])
			aug_mask = torch.Tensor(aug["mask"])
			return aug_img, aug_mask#torch.unsqueeze(aug_img, 0), torch.unsqueeze(aug_mask, 0)
		
		#int(img.shape, mask.shape)
		img = torch.Tensor(img)
		mask = torch.Tensor(mask)
		#print(img.shape, mask.shape)
		return img, mask
