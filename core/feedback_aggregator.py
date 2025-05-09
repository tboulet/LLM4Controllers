# Logging
from collections import defaultdict
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

from core.error_trace import ErrorTrace
from core.task import TaskDescription


class FeedbackType(Enum):
    BOOLEAN = "boolean"
    NUMERICAL = "numerical"
    ERROR = "error"


class MetricData:
    """A class to store a certain metric over time.
    
    It can either be :
        - a boolean (True/False), in this case the metric is storing the number of True and False values
            metric.n_true : int
            metric.n_false : int
            The sum = n_episodes
        - a numerical value (int/float), in this case the metric is storing the values in a list
            metric.values : List[float]
            The length of the list = n_episodes
        - an error message (ErrorTrace), in this case the metric is storing the error messages in a dictionary mapping error messages to their count
            metric.dictionary : Dict[str, int]        
    """

    def __init__(self):
        # Parameters
        self.type_value = None
        # Define the aggregator attributes
        self.values = []
        self.dictionary = defaultdict(int)
        self.n_true = 0
        self.n_false = 0

    def add(self, value: Any):
        """
        Add a value to the aggregator.

        Args:
            value (Any): The value to add.
        """
        # Check the type of the value and set the type_value accordingly
        if isinstance(value, bool):
            type_value = FeedbackType.BOOLEAN
            self.check_type_value(type_value)
            if value:
                self.n_true += 1
            else:
                self.n_false += 1
        elif isinstance(value, (int, float, np.float32, np.float64)):
            type_value = FeedbackType.NUMERICAL
            self.check_type_value(type_value)
            self.values.append(value)
        elif isinstance(value, ErrorTrace):
            type_value = FeedbackType.ERROR
            self.check_type_value(type_value)
            self.dictionary[value.error_message] += 1
        else:
            raise ValueError(
                f"Unsupported feedback type for value: {value} : {type(value)}"
            )

    def check_type_value(self, type_value: FeedbackType):
        """Check if the type_value is already set and matches the current value.
        If no type_value is set, set it to the current value.
        
        Args:
            type_value (FeedbackType): The type of the value to check.
        """
        if self.type_value is None:
            self.type_value = type_value
        else:
            if self.type_value != type_value:
                raise ValueError(
                    f"Feedback type mismatch: expected {self.type_value}, got {type_value}"
                )


class FeedbackAggregated:
    """
    A class to aggregate feedback over time.

    The feedback can be numerical, categorical, or textual.
    """

    def __init__(self):
        self.metrics: Dict[str, MetricData] = defaultdict(MetricData)
        self.n_episodes = 0
        self.dict_aggregated_feedback = None  # None while feedback is not aggregated

    def add_feedback(self, feedback: Dict[str, Any]):
        """
        Add feedback to the aggregator.

        Args:
            feedback (Dict[str, Any]): A dictionary containing feedback data.
        """
        for key, value in feedback.items():
            self.metrics[key].add(value)
        self.n_episodes += 1

    def aggregate(self):
        """
        Aggregate the feedback data.
        """
        assert (
            self.n_episodes > 0
        ), "No episodes feedback have been added to the feedback aggregator."
        dict_aggregated_feedback = {"n_episodes": self.n_episodes}
        for key, metric in self.metrics.items():
            if metric.type_value == FeedbackType.NUMERICAL:
                dict_aggregated_feedback[key] = {
                    "mean": np.mean(metric.values),
                    "std": np.std(metric.values),
                    "min": np.min(metric.values),
                    "max": np.max(metric.values),
                }
            elif metric.type_value == FeedbackType.BOOLEAN:
                assert (
                    metric.n_true + metric.n_false == self.n_episodes
                ), f"Number of true and false values do not match the number of episodes: {metric.n_true + metric.n_false} != {self.n_episodes}"
                dict_aggregated_feedback[key] = metric.n_true / self.n_episodes
            elif metric.type_value == FeedbackType.ERROR:
                dict_aggregated_feedback[key] = dict(metric.dictionary)
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        self.dict_aggregated_feedback = dict_aggregated_feedback

    def get_repr(self):
        """
        Get a string representation of the feedback aggregator.

        Returns:
            str: A string representation of the feedback aggregator.
        """
        assert (
            self.dict_aggregated_feedback is not None
        ), "Feedback has not been aggregated yet. Please call aggregate() first."
        list_repr = []
        for key, metric in self.metrics.items():
            # Numerical : mean, std, min, max
            if metric.type_value == FeedbackType.NUMERICAL:
                d = self.dict_aggregated_feedback[key]
                if self.n_episodes > 1:
                    list_repr.append(
                        f"{key} : {d['mean']:.2f} +/- {d['std']:.2f} (min: {d['min']:.2f}, max: {d['max']:.2f}, aggregated over {self.n_episodes} episodes)"
                    )
                else:
                    list_repr.append(f"{key} : {d['mean']:.2f}")
            # Boolean : percentage
            elif metric.type_value == FeedbackType.BOOLEAN:
                percentage = (
                    self.dict_aggregated_feedback[key] * 100
                )
                list_repr.append(
                    f"{key} rate : {percentage:.2f}% (aggregated over {self.n_episodes} episodes)"
                )
            # Error : percentage of each error
            elif metric.type_value == FeedbackType.ERROR:
                d : Dict[str, int] = self.dict_aggregated_feedback[key]
                repr_error = f"{key} : \n\t"
                list_repr_error = []
                for error_message, count in d.items():
                    percentage = (count / self.n_episodes) * 100
                    list_repr_error.append(
                        f"Happened {percentage:.2f}% of episodes : {error_message}"
                    )
                repr_error += "\n\t".join(list_repr_error)
                list_repr.append(repr_error)
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        return "\n".join(list_repr)
    
    def get_metrics(self, task : TaskDescription = None) -> Dict[str, Any]:
        """
        Get the aggregated feedback as loggable metrics.

        Args:
            task (TaskDescription, optional): The task description. Defaults to None.
            If provided, the metrics will be prefixed with the task name except for the error metrics.
            
        Returns:
            Dict[str, Any]: A dictionary containing the aggregated feedback metrics.
        """
        assert (
            self.dict_aggregated_feedback is not None
        ), "Feedback has not been aggregated yet. Please call aggregate() first."
        metrics = {}
        for key, metric in self.metrics.items():
            # Numerical : mean, std, min, max
            if metric.type_value == FeedbackType.NUMERICAL:
                d = self.dict_aggregated_feedback[key]
                metrics[f"{key}_mean"] = d["mean"]
                # metrics[f"{key}_std"] = d["std"]
                # metrics[f"{key}_min"] = d["min"]
                # metrics[f"{key}_max"] = d["max"]
            # Boolean : rate
            elif metric.type_value == FeedbackType.BOOLEAN:
                metrics[f"{key}_rate"] = (
                    self.dict_aggregated_feedback[key]
                )
            # Error : rate of each error
            elif metric.type_value == FeedbackType.ERROR:
                if task is not None:
                    continue # skip the error metrics if task is provided
                d : Dict[str, int] = self.dict_aggregated_feedback[key]
                for error_message, count in d.items():
                    error_rate = count / self.n_episodes
                    metrics[f"{key}_{error_message}_rate"] = error_rate
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        # Add task specific metrics
        if task is not None:
            metrics = {
                f"{task}_{key}": value
                for key, value in metrics.items()
            }
        # Add the number of episodes
        metrics["n_episodes"] = self.n_episodes
        # Return the metrics
        return metrics

    def __repr__(self):
        """
        Get a string representation of the feedback aggregator.

        Returns:
            str: A string representation of the feedback aggregator.
        """
        return self.get_repr()