import logging
import os 
from src.common.path_setup import *

LOGGER_NAME="DATA_NLP_VISION_EXERCISE"
LOGGING_LEVEL = logging.INFO

log_file = os.path.join(logs_dir, "app.log")
logging.basicConfig(filename=log_file, level=LOGGING_LEVEL,format="%(levelname)s:%(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(LOGGER_NAME)