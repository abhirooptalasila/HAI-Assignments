import logger
log = logger.setup_logger()

import torch
from unet.models import UNet
from torchsummary import summary

log.debug('Calling module function.')

image = torch.rand((1, 1, 512, 512))
model = UNet(1, 2)
x = model(image)
# log.debug(summary(model, (1, 512, 512)))
del model 
del image
# log.warning()
log.debug('Finished.')