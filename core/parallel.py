from collections import defaultdict
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
import os
import re
import signal
import time
from typing import Any, Callable, Dict, Iterator, List
from tbutils.tmeasure import RuntimeMeter, get_runtime_metrics
import tiktoken

from core.utils import get_error_info


def get_chunks(lst: List[Any], size_chunk: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), size_chunk):
        yield lst[i : i + size_chunk]


def run_parallel(
    func: Callable,
    batch_configs: List[Dict],
    config_parallel: Dict,
    return_value_on_exception: Any = None,
) -> List[Any]:
    """
    Run a function in parallel using ProcessPoolExecutor.

    Args:
        func (Callable): The function to run in parallel.
        batch_configs (List[Dict]): A list of dictionaries containing the configurations for each run.
        config_parallel (Dict): Configuration for parallel execution, including stuff like the number of workers.
        return_value_exception (Any): The value to return in case of an exception.

    Returns:
        List[Any]: A list of results from the function calls.
    """
    futures = []
    method = config_parallel["method"]

    # ThreadPoolExecutor
    if method == "thread_pool":
        results = []
        max_workers = config_parallel["max_workers"]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sub_batch_configs_tasks in get_chunks(batch_configs, max_workers):
                for config_task in sub_batch_configs_tasks:
                    future = executor.submit(func, **config_task)
                    futures.append(future)
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                print(f"Exception in thread: {get_error_info(e, string_mode=False)}")
                result = return_value_on_exception
            results.append(result)
        return results

    # Simple sequential
    elif method == "sequential":
        results = []
        for config_task in batch_configs:
            try:
                result = func(**config_task)
            except Exception as e:
                print(f"Exception in thread: {get_error_info(e, string_mode=False)}")
                result = return_value_on_exception
            results.append(result)
        return results

    else:
        raise ValueError(f"Unknown method: {method}")
