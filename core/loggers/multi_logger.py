from typing import Dict, List, Tuple, Type, Union

import numpy as np

from core.utils import one_time_warning
from .base_logger import BaseLogger
from tensorboardX import SummaryWriter

class MultiLogger(BaseLogger):
    
    def __init__(self, *loggers : BaseLogger):
        self.loggers = loggers
        
    def log_scalars(self, metrics, step):
        for logger in self.loggers:
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
    
    