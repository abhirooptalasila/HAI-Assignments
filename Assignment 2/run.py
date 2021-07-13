import logger
log = logger.setup_logger()

import torch
import numpy as np
from unet.utils import *

def train(model, train_dl, val_dl, ds=False, epochs=500):
	#model = UNet(1, 1)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if device.type == "cuda":
		model.cuda()
		model = model.type(torch.cuda.FloatTensor)

	criterion = torch.nn.BCEWithLogitsLoss()
	#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
	log.info("Started training for {} for {} epochs".format(
		model.__class__.__name__, epochs))

	for e in range(epochs):
		train_loss = 0.0
		for data, labels in train_dl:
			if device.type == "cuda":
				data = data.type(torch.cuda.FloatTensor)
				labels = labels.type(torch.cuda.FloatTensor)
				data, labels = data.cuda(), labels.cuda()

			if ds:
				outputs = model(data)
				loss = 0
				for output in outputs:
					loss += criterion(output, labels)
				loss /= len(outputs)
			else:
				target = model(data)
				#print(data.shape, target.shape, labels.shape, target, torch.sigmoid(target))
				loss = criterion(target, labels)
			
			train_loss = loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		valid_loss = validation(model, device, val_dl, criterion, ds)

		del data, labels; torch.cuda.empty_cache()

		if e % 50 == 0:
			log.debug('Epoch {} \t Training Loss: {} \t Validation Loss: {}'.format(
					e+1, train_loss, valid_loss))
			validation(model, device, val_dl, criterion, ds, True)

		# if min_valid_loss > valid_loss:
		# 	min_valid_loss = valid_loss
		# 	# Saving State Dict
		# 	torch.save({
		# 			'model_state_dict': model.state_dict(),
		# 			'optimizer_state_dict': optimizer.state_dict(),
		# 			}, save_path)


def validation(model, device, dl, lossfn, ds, metrics=False):
	val_loss, mean_iou, mean_dice = 0.0, [], []
	model.eval()
	with torch.no_grad():
		for val_data, val_labels in dl:
			if device.type == "cuda":
				val_data = val_data.type(torch.cuda.FloatTensor)
				val_labels = val_labels.type(torch.cuda.FloatTensor)
				val_data, val_labels = val_data.cuda(), val_labels.cuda()

			if ds:
				outputs = model(val_data)
				loss = 0
				for output in outputs:
					loss += lossfn(output, val_labels)
				loss /= len(outputs)
				iou = iou_score(outputs[-1], val_labels)
				dice = dice_coef(outputs[-1], val_labels)
			else:
				target = model(val_data)
				loss = lossfn(target, val_labels)
				iou = iou_score(target, val_labels)
				dice = dice_coef(target, val_labels)
			
			val_loss = loss.item()
			mean_iou.append(iou); mean_dice.append(dice)
	
	del val_data, val_labels
	if metrics:
		log.debug('Mean IOU: {} \t Mean Dice Coef: {}'.format(
			np.mean(mean_iou), np.mean(mean_dice)))
		return 

	return val_loss

if __name__ == "__main__":
	pass
