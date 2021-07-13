import os
import torch
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt


def visualize(image, mask):
	f, ax = plt.subplots(1, 2, figsize=(8, 8))
	ax[0].imshow(image, cmap="bone")
	ax[1].imshow(mask, cmap="bone")

def normalize_image(image):
	img = Image.open(image)
	img_arr = np.asarray(img) / 255
	return img_arr

def images_mean(image_foler):
	base = os.path.abspath(image_foler)
	files = os.listdir(base)
	mean_sum = 0
	for img in files:
		img_arr = normalize_image(os.path.join(base, img))
		mean_sum += np.mean(img_arr)
	return mean_sum / len(files)

def images_std(image_foler):
	base = os.path.abspath(image_foler)
	files = os.listdir(base)
	std_sum = 0
	for img in files:
		img_arr = normalize_image(os.path.join(base, img))
		std_sum += np.std(img_arr)
	return std_sum / len(files)

def process_ISBI(imgs, masks, path):
	"""Process .tiff slices into individual files
	Don't run more than once"""
	img_count = imgs.shape[0]
	val_count = int(img_count*0.2)

	train_dir = os.path.join(path, "train")
	val_dir = os.path.join(path, "val")

	for x in range(img_count - val_count):
		ipath = os.path.join(train_dir, "images")
		os.makedirs(ipath, exist_ok=True)
		im = Image.fromarray(imgs[x])
		im.save(os.path.join(ipath, str(x) + ".png"))
		
		mpath = os.path.join(train_dir, "masks")
		os.makedirs(mpath, exist_ok=True)
		im = Image.fromarray(masks[x])
		im.save(os.path.join(mpath, str(x) + ".png"))
		
	for x in range(1, val_count+1):
		ipath = os.path.join(val_dir, "images")
		os.makedirs(ipath, exist_ok=True)
		im = Image.fromarray(imgs[-x])
		im.save(os.path.join(ipath, str(x) + ".png"))
		
		mpath = os.path.join(val_dir, "masks")
		os.makedirs(mpath, exist_ok=True)
		im = Image.fromarray(masks[-x])
		im.save(os.path.join(mpath, str(x) + ".png"))

# def iou_score(output, target):
#     smooth = 1e-5

#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()

#     return (intersection + smooth) / (union + smooth)

def iou_score(outputs, labels):
	EPS = 1e-6
	labels = labels.cpu().int()
	pred = torch.sigmoid(outputs)
	pred[pred > 0.5] = 1; pred[pred <= 0.5] = 0
	pred = pred.cpu().detach().int()

	# Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
	intersection = (pred & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
	union = (pred | labels).float().sum((1, 2))  # Will be zero if both are 0

	iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

	# thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
	# return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
	return iou.mean()


def dice_coef(output, target):
	smooth = 1e-5

	output = torch.sigmoid(output).view(-1).data.cpu().numpy()
	target = target.view(-1).data.cpu().numpy()
	intersection = (output * target).sum()

	return (2. * intersection + smooth) / \
		(output.sum() + target.sum() + smooth)

def predict_masks(models, dataloader, count=5):
	with torch.set_grad_enabled(False):
		for data, labels in dataloader:
			if count <= 0:
				return
			orig_image = data.cpu().detach().numpy()[0][0]
			orig_mask = labels.cpu().detach().numpy()[0][0]

			output = models[0](data.cuda())
			unet_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]

			output = models[1](data.cuda())
			wunet_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]

			output = models[3](data.cuda())[-1]
			unetpds_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]
			unetpds_p[unetpds_p > 0.5] = 1; unetpds_p[unetpds_p <= 0.5] = 0

			output = models[-1](data.cuda())[-1]
			unetppds_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]
			unetppds_p[unetppds_p > 0.5] = 1; unetppds_p[unetppds_p <= 0.5] = 0
			
			fig, _ = plt.subplots(1, 6, figsize=(16, 12))
			ops = [orig_image, orig_mask, unet_p, wunet_p, unetpds_p, unetppds_p]
			for i, op in enumerate(ops):
				fig.axes[i].imshow(op, cmap='gray')
				fig.axes[i].get_xaxis().set_visible(False)
				fig.axes[i].get_yaxis().set_visible(False)
			plt.show();
			count -= 1

def plot_masks(models, dataloader, count=5):
	with torch.set_grad_enabled(False):
		for data, labels in dataloader:
			if count <= 0:
				return
			orig_image = data.cpu().detach().numpy()[0][0]
			orig_mask = labels.cpu().detach().numpy()[0][0]

			output = models[0](data.cuda())
			unet_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]

			output = models[1](data.cuda())
			wunet_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]

			output = models[3](data.cuda())
			unetpds_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]
			unetpds_p[unetpds_p > 0.5] = 1; unetpds_p[unetpds_p <= 0.5] = 0

			output = models[-1](data.cuda())[-1]
			unetppds_p = torch.sigmoid(output).cpu().detach().numpy()[0][0]
			unetppds_p[unetppds_p > 0.5] = 1; unetppds_p[unetppds_p <= 0.5] = 0
			
			fig, _ = plt.subplots(1, 6, figsize=(16, 12))
			ops = [orig_image, orig_mask, unet_p, wunet_p, unetpds_p, unetppds_p]
			for i, op in enumerate(ops):
				fig.axes[i].imshow(op, cmap='gray')
				fig.axes[i].get_xaxis().set_visible(False)
				fig.axes[i].get_yaxis().set_visible(False)
			plt.show();
			count -= 1