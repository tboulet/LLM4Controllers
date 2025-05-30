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

from core.types import ErrorTrace, TextualInformation
from core.task import TaskDescription
from core.utils import sanitize_name

def join_texts_under_limit(text_list, n_text_chars_max):
    """
    Randomly joins texts from the list with '\n\n' so the total length stays under n_text_chars_max.
    If the first randomized text exceeds the limit, returns that one alone.
    
    Parameters:
        text_list (list of str): List of text segments.
        n_text_chars_max (int): Max total character count.
    
    Returns:
        str: Concatenated texts separated by '\n\n'.
    """
    if len(text_list) == 0:
        raise ValueError("The text list cannot be empty.")
    if len(text_list) == 1:
        return text_list[0]
    
    shuffled = text_list.copy()
    random.shuffle(shuffled)

    joined_texts = []
    total_chars = 0

    for text in shuffled:
        next_len = len(text) if not joined_texts else len(text) + 2  # +2 for "\n\n"
        if total_chars + next_len > n_text_chars_max:
            if not joined_texts:
                return text  # First text alone exceeds limit
            break
        joined_texts.append(f"Text: {text}")
        total_chars += next_len

    res =  "\n\n".join(joined_texts)
    
    return f"{len(joined_texts)} texts extracted from the aggregated list of texts:\n\n{res}"


def mean_all_tensor_elements(list_tensor):
    """
    Compute the mean across a list of tensors for each element in the full index space.

    Parameters:
    - list_tensor: List of NumPy arrays, all of the same shape (n1, n2, ..., nN)

    Returns:
    - Dictionary: keys are 'mean_i0_i1_..._iN' and values are mean values at those positions.
    """
    stacked = np.stack(list_tensor)  # shape: (num_tensors, n1, n2, ..., nN)
    mean_tensor = np.mean(stacked, axis=0)  # shape: (n1, n2, ..., nN)

    result = {
        f"mean_{'_'.join(map(str, idx))}": mean_tensor[idx]
        for idx in np.ndindex(mean_tensor.shape)
    }
    return result


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

    # If values only contains 0 and 1, go back to pass_at_k
    if all(v in [0, 1] for v in values):
        return pass_at_k([bool(v) for v in values], k)

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


def function_names_to_agg_functions(agg_name: str):
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
    elif re.match(r"worst@(\d+)", agg_name):
        k = int(re.match(r"worst@(\d+)", agg_name).group(1))
        return lambda values: -best_at_k([-v for v in values], k)
    else:
        raise ValueError(
            f"Unsupported aggregation function: {agg_name}. Supported functions are: mean, median, std, min, max, pass@k, best@k."
            " For pass@k and best@k, use the format 'pass@<k>' or 'best@<k>'."
        )


class FeedbackType(Enum):
    BOOLEAN = "boolean"
    NUMERICAL = "numerical"
    ERROR = "error"
    TENSOR = "tensor"
    TEXTUAL = "textual"


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
        - a tensor shaped numpy arrat (np.ndarray), in this case the metric is storing the tensors in a list
            metric.tensor_values : List[np.ndarray]
        - a textual information (TextualInformation), in this case the metric is storing the texts in a list
            metric.texts : List[TextualInformation]
    """

    def __init__(self):
        # Parameters
        self.type_value = None
    
    def init_aggregator(self):
        # Define the aggregator attributes
        if self.type_value == FeedbackType.BOOLEAN:
            self.n_true : int = 0
            self.n_false : int = 0
        elif self.type_value == FeedbackType.NUMERICAL:
            self.values : List[float] = []
        elif self.type_value == FeedbackType.ERROR:
            self.dictionary = defaultdict(int)
        elif self.type_value == FeedbackType.TENSOR:
            self.tensor_values : List[np.ndarray] = []
        elif self.type_value == FeedbackType.TEXTUAL:
            self.texts : List[TextualInformation] = []
        else:
            raise ValueError(f"Unsupported feedback type: {self.type_value}")

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
        elif isinstance(value, np.ndarray):
            type_value = FeedbackType.TENSOR
            self.check_type_value(type_value)
            self.tensor_values.append(value)
        elif isinstance(value, TextualInformation):
            type_value = FeedbackType.TEXTUAL
            self.check_type_value(type_value)
            self.texts.append(value)
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
            self.init_aggregator()
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

    def __init__(self, agg_methods: List[str] = ["mean"], n_text_chars_max : int = 1000):
        self.metrics: Dict[str, MetricData] = defaultdict(MetricData)
        self.n_data = 0
        self.dict_aggregated_feedback = None  # None while feedback is not aggregated
        self.agg_methods = agg_methods
        self.n_text_chars_max = n_text_chars_max  # Maximum number of characters to keep in textual feedback

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
                if len(metric.values) < self.n_data:
                    metric.values += [0] * (self.n_data - len(metric.values))
                dict_aggregated_feedback[key] = {
                    agg_name: function_names_to_agg_functions(agg_name)(metric.values)
                    for agg_name in self.agg_methods
                }
            elif metric.type_value == FeedbackType.BOOLEAN:
                if metric.n_true + metric.n_false < self.n_data:
                    metric.n_false += self.n_data - metric.n_true - metric.n_false
                assert (
                    metric.n_true + metric.n_false == self.n_data
                ), f"Number of true and false values do not match the number of datapoints: {metric.n_true + metric.n_false} != {self.n_data}"
                dict_aggregated_feedback[key] = metric.n_true / self.n_data
            elif metric.type_value == FeedbackType.ERROR:
                dict_aggregated_feedback[key] = dict(metric.dictionary)
            elif metric.type_value == FeedbackType.TENSOR:
                indexes_to_mean = mean_all_tensor_elements(metric.tensor_values)
                dict_aggregated_feedback[key] = indexes_to_mean
            elif metric.type_value == FeedbackType.TEXTUAL:
                dict_aggregated_feedback[key] = metric.texts
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
            elif metric.type_value == FeedbackType.TENSOR:
                d = self.dict_aggregated_feedback[key]
                repr_tensor = "\n".join([
                    f"{key}_{index} : {value:.2f} (aggregated over {self.n_data} datapoints)" for index, value in d.items()
                ])
                list_repr.append(repr_tensor)
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
                    error_message = error_message.replace("\n", "\n\t")
                    list_repr_error.append(
                        f"Happened {percentage:.2f}% of time : {error_message}"
                    )
                repr_error += "\n\t".join(list_repr_error)
                list_repr.append(repr_error)
            # Textual : list of texts
            elif metric.type_value == FeedbackType.TEXTUAL:
                texts : List[str] = self.dict_aggregated_feedback[key]
                list_repr.append(f"{key} : {join_texts_under_limit(texts, self.n_text_chars_max)}")
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        return "\n".join(list_repr)

    def get_metrics(
        self,
        prefix: TaskDescription = None,
        do_log_no_prefix: bool = True,
    ) -> Dict[str, Any]:
        """
        Get the aggregated feedback as loggable metrics.

        Args:
            prefix (TaskDescription, optional): The task name. Defaults to None. If provided, the metrics will be prefixed with the task name.
            do_log_no_prefix (bool, optional): If True, the metrics will also be logged without prefix. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the aggregated feedback metrics.
        """
        assert (
            self.dict_aggregated_feedback is not None
        ), "Feedback has not been aggregated yet. Please call aggregate() first."
        assert (
            prefix is not None or do_log_no_prefix
        ), "Either prefix or do_log_no_prefix should be provided."
        metrics = {}
        for key, metric in self.metrics.items():
            # Numerical : mean, std, min, max
            if metric.type_value == FeedbackType.NUMERICAL:
                d = self.dict_aggregated_feedback[key]
                metrics.update(
                    {f"{key}/{agg_name}": d[agg_name] for agg_name in self.agg_methods}
                )
            # Tensor : mean of each index
            elif metric.type_value == FeedbackType.TENSOR:
                d = self.dict_aggregated_feedback[key]
                for index, value in d.items():
                    metrics[f"{key}_{index}"] = value
            # Boolean : rate
            elif metric.type_value == FeedbackType.BOOLEAN:
                metrics[f"{key}/rate"] = self.dict_aggregated_feedback[key]
            # Error : rate of each error
            elif metric.type_value == FeedbackType.ERROR:
                # if task is not None:
                #     continue # skip the error metrics if task is provided
                d: Dict[str, int] = self.dict_aggregated_feedback[key]
                for error_message, count in d.items():
                    error_rate = count / self.n_data
                    metrics[f"{key}_{error_message}/rate"] = error_rate
            # Textual : dont log
            elif metric.type_value == FeedbackType.TEXTUAL:
                texts: List[TextualInformation] = self.dict_aggregated_feedback[key]
                metrics[f"{key}/len_text_average"] = np.mean(
                    [len(text.text) for text in texts]
                ) if texts else 0
            else:
                raise ValueError(f"Unsupported feedback type: {metric.type_value}")
        # Log what is needed
        metrics_res = {}
        if prefix is not None:
            prefix = sanitize_name(prefix)
            metrics_res.update(
                {f"{prefix}/{key}": value for key, value in metrics.items()}
            )
        if do_log_no_prefix:
            metrics_res.update(metrics)
        # Return the metrics
        return metrics_res

    def __repr__(self):
        """
        Get a string representation of the feedback aggregator.

        Returns:
            str: A string representation of the feedback aggregator.
        """
        return self.get_repr()
