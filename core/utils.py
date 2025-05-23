import time
import traceback
from typing import Dict, Type, Any, Tuple, Union
import numpy as np
import signal
from contextlib import contextmanager


def get_error_info(e):
    stack_trace = traceback.format_exc()  # Capture the full stack trace as a string
    relevant_stack_trace = "\n".join(
        line for line in stack_trace.splitlines() if "<string>" in line
    )
    error_info = f"Stack Trace:\n{relevant_stack_trace} \nError Output:\n{e}"
    return error_info


dict_warning_messages = set()


def one_time_warning(warning_message):
    if warning_message not in dict_warning_messages:
        print(f"WARNING : {warning_message}")
        dict_warning_messages.add(warning_message)


def try_get_seed(config: Dict) -> int:
    """Will try to extract the seed from the config, or return a random one if not found

    Args:
        config (Dict): the run config

    Returns:
        int: the seed
    """
    try:
        seed = config["seed"]
        if not isinstance(seed, int):
            seed = np.random.randint(0, 1000)
    except KeyError:
        seed = np.random.randint(0, 1000)
    return seed


def to_maybe_inf(n):
    if n in ["inf", "infinity", None, "None"]:
        return np.inf
    return n


import os
import re


def get_name_copy(path: str) -> str:
    """Generate a unique filename by appending a number to the base name if it already exists.
    Examples :
        test.txt -> test_2.txt
        test_6.txt -> test_7.txt
        test_6_2.txt -> test_6_3.txt

    Args:
        path (str): the path to the file

    Returns:
        str: a different path for the file, with a number appended to the base name
    """
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)

    # Match filename ending in _number (e.g., video_6)
    match = re.match(r"^(.*)_(\d+)$", name)
    if match:
        base, num = match.groups()
        num = int(num) + 1
    else:
        base = name
        num = 2

    new_filename = f"{base}_{num}{ext}"
    new_path = os.path.join(directory, new_filename)

    return new_path


def sanitize_name(name: str) -> str:
    name = name.replace('/', ' div ')
    name = name.replace(' ', '_')
    name = name.replace('(', '')
    name = name.replace(')', '')
    return name