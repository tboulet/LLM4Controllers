from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

from tbutils.exec_max_n import print_once

class BaseLogger(ABC):
    """Base class for all loggers"""

    def log_scalars(
        self,
        metrics: Dict[str, Union[float, int]],
        step: int,
    ):
        """Log dictionary of scalars"""
        pass

    def log_histograms(
        self,
        histograms: Dict[str, List[float]],
        step: int,
    ):
        """Log dictionary of histograms"""
        print_once(f"WARNING : {self.__class__.__name__} does not support logging of histograms")
    
    def log_images(
        self,
        images: Dict[str, List[List[float]]],
        step: int,
    ):
        """Log dictionary of maps"""
        print_once(f"WARNING : {self.__class__.__name__} does not support logging of images")

    def log_info(
        self,
        info: Dict[str, Any],
        step: int,
    ):
        """Log dictionary of info"""
        print_once(f"WARNING : {self.__class__.__name__} does not support logging of info")
        
    def close(self):
        """Close the logger"""
        pass