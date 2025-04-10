# Logging
import os
import shutil
import sys
import wandb
from tensorboardX import SummaryWriter

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# Utils
from tqdm import tqdm
import datetime
from time import time, sleep
from typing import Any, Dict, Type
import cProfile

# ML libraries
import random
import numpy as np
import torch
import transformers

from enum import Enum

class FeedbackType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXTUAL = "textual"

class MetricAggregator:
    """A class to aggregate a certain metric over time.
    """
    def __init__(self, percentage : bool = False):
        self.values = []
        self.type = None
        
    def add(self, value: Any):
        """
        Add a value to the aggregator.
        
        Args:
            value (Any): The value to add.
        """
        if isinstance(value, (int, float, np.ndarray)):
            type_value = FeedbackType.NUMERICAL
        elif isinstance(value, str):
            type_value = FeedbackType.TEXTUAL
        elif isinstance(value, bool):
            type_value = FeedbackType.CATEGORICAL
        else:
            raise ValueError(f"Unsupported feedback type for value: {value} : {type(value)}")
        
        if self.type is None:
            self.type = type_value
        else:
            if self.type != type_value:
                raise ValueError(f"Feedback type mismatch: expected {self.type}, got {type_value}")
            
        self.values.append(value)
        
    def aggregate
    
class FeedbackAggregator:
    """
    A class to aggregate feedback over time.
    
    The feedback can be numerical, categorical, or textual.
    """
    def __init__(self):
        self.metrics : Dict[str, MetricAggregator] = {}
    
    def add_feedback(self, feedback: Dict[str, Any]):
        """
        Add feedback to the aggregator.
        
        Args:
            feedback (Dict[str, Any]): A dictionary containing feedback data.
        """
        for key, value in feedback.items():
            if key not in self.metrics:
                self.metrics[key] = MetricAggregator()
            self.metrics[key].add(value)