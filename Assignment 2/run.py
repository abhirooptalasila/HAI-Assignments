import logger
log = logger.setup_logger(file_name = 'app_debug.log')

import unet # as unet

log.debug('Calling module function.')
log.warning(unet.add(5, 2))
log.debug('Finished.')