import logger
log = logger.get_logger(__name__)

def add(num1, num2):
    log.debug("Executing add function.")
    return num1 + num2