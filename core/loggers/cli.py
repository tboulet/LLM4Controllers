from typing import Dict, List, Tuple, Type, Union
from .base_logger import BaseLogger

class LoggerCLI(BaseLogger):
    
    def log_scalars(self, metrics, step):
        if len(metrics) == 0:
            return
        message = f"Step {step} :\n"
        message += "\n".join([f"\t{key} : {value}" for key, value in metrics.items()])
        print(message)