from typing import Dict, List, Tuple, Type, Union

import numpy as np

from core.utils import one_time_warning
from .base_logger import BaseLogger
from tensorboardX import SummaryWriter

class LoggerTensorboard(BaseLogger):
    
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        
    def log_scalars(self, metrics, step):
        if len(metrics) == 0:
            return
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_histograms(
        self,
        histograms: Dict[str, List[float]],
        step: int,
    ):
        for name, values in histograms.items():
            if len(values) > 0:
                try:
                    self.writer.add_histogram(name, values, step)
                except:
                    one_time_warning(f"Error in logging histogram for class {self.__class__.__name__}", additional_message=f"The error was for name={name}, values={values}, timestep={step}")
    
    def log_maps(
        self,
        images: Dict[str, List[List[float]]],
        step: int,
    ):
        for name, map in images.items():
            map = np.array(map)
            image = np.zeros((3, map.shape[0], map.shape[1]))
            highest_value = map.max()
            if highest_value > 0:  
                image[2, map > 0] = map[map > 0] / highest_value
            lowest_value = map.min()
            if lowest_value < 0:
                image[0, map < 0] = -map[map < 0] / -lowest_value
            self.writer.add_image(name, image, step)
        
    def close(self):
        self.writer.close()
    
    