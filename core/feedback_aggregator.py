# Logging
from collections import defaultdict
import os
import re
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
from typing import Any, Callable, Dict, List, Type
import cProfile
from enum import Enum

# ML libraries
import random
import numpy as np
import torch
import transformers
from math import comb
from itertools import combinations

from core.error_trace import ErrorTrace
from core.task import TaskDescription


def pass_at_k(values: List[bool], k: int) -> float:
    """
    Estimate pass@k given a list of boolean values indicating correctness of each sample.

    Args:
        values (List[bool]): List of pass/fail outcomes (True = correct).
        k (int): Number of samples to consider.

    Returns:
        float: Estimated probability that at least one of k samples is correct.
    """
    assert all(isinstance(v, bool) for v in values), "All values must be boolean"
    assert 0 <= k <= len(values), "k must be between 0 and the number of samples"
    n = len(values)
    c = sum(values)  # number of correct samples
    if c == 0:
        return 0.0  # no correct solutions at all
    if c == n:
        return 1.0  # all samples are correct
    return 1.0 - comb(n - c, k) / comb(n, k)


def best_at_k(values: List[float], k: int, num_samples: int = 1000) -> float:
    """
    Estimate best@k: the expected maximum score among k samples drawn without replacement.

    Args:
        values (List[float]): List of score values in [0, 1].
        k (int): Number of samples to draw.
        num_samples (int): Number of random subsets to sample for estimation (used when n choose k is large).

    Returns:
        float: Estimated expected maximum score from k samples.
    """
    assert 0 < k <= len(values), "k must be between 1 and the number of samples"
    
    n = len(values)
    
    # Use full enumeration if tractable
    total_combinations = comb(n, k)
    if total_combinations <= num_samples:
        subsets = combinations(values, k)
    else:
        # Use random sampling
        subsets = (random.sample(values, k) for _ in range(num_samples))

    maxima = [max(subset) for subset in subsets]
    return float(np.mean(maxima))


def function_names_to_agg_functions(agg_name : str):
    if agg_name == "mean":
        return np.mean
    elif agg_name == "median":
        return np.median
    elif agg_name == "std":
        return np.std
    elif agg_name == "min":
        return np.min
    elif agg_name == "max":
        return np.max
    elif re.match(r"pass@(\d+)", agg_name):
        k = int(re.match(r"pass@(\d+)", agg_name).group(1))
        return lambda values: pass_at_k(values, k)
    elif re.match(r"best@(\d+)", agg_name):
        k = int(re.match(r"best@(\d+)", agg_name).group(1))
        return lambda values: best_at_k(values, k)
    else:
        raise ValueError(
            f"Unsupported aggregation function: {agg_name}. Supported functions are: mean, median, std, min, max, pass@k, best@k."
            " For pass@k and best@k, use the format 'pass@<k>' or 'best@<k>'."
        )


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
            The sum = n_data
        - a numerical value (int/float), in this case the metric is storing the values in a list
            metric.values : List[float]
            The length of the list = n_data
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

    def __init__(self, agg_methods: List[str] = ["mean"]):
        self.metrics: Dict[str, MetricData] = defaultdict(MetricData)
        self.n_data = 0
        self.dict_aggregated_feedback = None  # None while feedback is not aggregated
        self.agg_methods = agg_methods

    def add_feedback(self, feedback: Dict[str, Any]):
        """
        Add feedback to the aggregator.

        Args:
            feedback (Dict[str, Any]): A dictionary containing feedback data.
        """
        for key, value in feedback.items():
            self.metrics[key].add(value)
        self.n_data += 1

    def aggregate(self):
        """
        Aggregate the feedback data.
        """
        assert self.n_data > 0, "No data have been added to the feedback aggregator."
        dict_aggregated_feedback = {}
        for key, metric in self.metrics.items():
            if metric.type_value == FeedbackType.NUMERICAL:
                dict_aggregated_feedback[key] = {
                    agg_name: function_names_to_agg_functions(agg_name)(metric.values)
                    for agg_name in self.agg_methods
                }
            elif metric.type_value == FeedbackType.BOOLEAN:
                assert (
                    metric.n_true + metric.n_false == self.n_data
                ), f"Number of true and false values do not match the number of datapoints: {metric.n_true + metric.n_false} != {self.n_data}"
                dict_aggregated_feedback[key] = metric.n_true / self.n_data
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
                repr_metric = f"{key} : {d['mean']:.2f}"
                if self.n_data == 1:
                    list_repr.append(repr_metric)
                    continue
                else:
                    if "std" in self.agg_methods:
                        repr_metric += f" +/- {d['std']:.2f}"
                    if "min" in self.agg_methods:
                        repr_metric += f", min: {d['min']:.2f}"
                    if "max" in self.agg_methods:
                        repr_metric += f", max: {d['max']:.2f}"
                    repr_metric += f" (aggregated over {self.n_data} datapoints)"
                    list_repr.append(repr_metric)
            # Boolean : percentage
            elif metric.type_value == FeedbackType.BOOLEAN:
                percentage = self.dict_aggregated_feedback[key] * 100
                list_repr.append(
                    f"{key} rate : {percentage:.2f}% (aggregated over {self.n_data} datapoints)"
                )
            # Error : percentage of each error
            elif metric.type_value == FeedbackType.ERROR:
                d: Dict[str, int] = self.dict_aggregated_feedback[key]
                repr_error = f"{key} : \n\t"
                list_repr_error = []
                for error_message, count in d.items():
                    percentage = (count / self.n_data) * 100
                    list_repr_error.append(
                        f"Happened {percentage:.2f}% of time : {error_message}"
                    )
                repr_error += "\n\t".join(list_repr_error)
                list_repr.append(repr_error)
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        return "\n".join(list_repr)

    def get_metrics(
        self,
        task: TaskDescription = None,
    ) -> Dict[str, Any]:
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
                metrics.update(
                    {
                        f"{key}_{agg_name}": d[agg_name]
                        for agg_name in self.agg_methods
                    }
                )
            # Boolean : rate
            elif metric.type_value == FeedbackType.BOOLEAN:
                metrics[f"{key}_rate"] = self.dict_aggregated_feedback[key]
            # Error : rate of each error
            elif metric.type_value == FeedbackType.ERROR:
                # if task is not None:
                #     continue # skip the error metrics if task is provided
                d: Dict[str, int] = self.dict_aggregated_feedback[key]
                for error_message, count in d.items():
                    error_rate = count / self.n_data
                    metrics[f"{key}_{error_message}_rate"] = error_rate
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        # Add task specific metrics
        if task is not None:
            metrics = {f"{task}_{key}": value for key, value in metrics.items()}
        # Return the metrics
        return metrics

    def __repr__(self):
        """
        Get a string representation of the feedback aggregator.

        Returns:
            str: A string representation of the feedback aggregator.
        """
        return self.get_repr()
