import logger
log = logger.setup_logger(file_name = 'app_debug.log')

from unet.models import UNet
import torch

log.debug('Calling module function.')
image = torch.rand((1, 1, 572, 572))
model = UNet()
model.forward(image)
# log.warning()
log.debug('Finished.')