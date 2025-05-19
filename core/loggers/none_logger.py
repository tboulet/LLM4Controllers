from typing import Dict, List, Tuple, Type, Union
from .base_logger import BaseLogger

class NoneLogger(BaseLogger):
    """A logger that does nothing."""
    def log_scalars(self, metrics, step):
        pass