import re
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from .base_logger import BaseLogger
from tensorboardX import SummaryWriter
from tbutils.tmeasure import RuntimeMeter
from tbutils.exec_max_n import print_once

unsafe_chars = r'[\\\s:.,;=?!&#@!%*"\'\[\]\{\}\(\)]'

class MultiLogger(BaseLogger):

    def __init__(self, *loggers: BaseLogger):
        self.loggers = loggers

    def log_scalars(self, metrics, step):
        for key in metrics:
            matches = re.search(unsafe_chars, key)
            if matches:
                print_once(f"WARNING : metric key '{key}' contains unsafe characters : {matches.group(0)}. This may cause issues with some loggers.")
        for logger in self.loggers:
            with RuntimeMeter(f"log_scalars_{logger.__class__.__name__}"):
                logger.log_scalars(metrics, step)

    def log_histograms(
        self,
        histograms: Dict[str, List[float]],
        step: int,
    ):
        for logger in self.loggers:
            logger.log_histograms(histograms, step)

    def log_images(
        self,
        images: Dict[str, List[List[float]]],
        step: int,
    ):
        for logger in self.loggers:
            logger.log_images(images, step)

    def close(self):
        for logger in self.loggers:
            logger.close()
