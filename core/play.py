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
import multiprocessing

# ML libraries
import random
import numpy as np
import torch
import transformers

# Project imports
from agent.base_controller import Controller
from core.error_trace import ErrorTrace
from core.feedback_aggregator import FeedbackAggregated

from core.task import Task
from core.types import ActionType
from core.utils import get_error_info
from core.register_hydra import register_hydra_resolvers


def act_time_bounded(controller: Controller, obs: Any, time_limit: float) -> ActionType:
    """Act using the controller with a time limit.

    Args:
        controller (Controller): The controller to use.
        obs (Any): The observation to use.
        time_limit (float, optional): The time limit for the action. Defaults to 2.0.

    Raises:
        TimeoutError: If the action takes too long.
        
    Returns:
        Any: The action to take.
    """
    # Use a multiprocessing pool with 1 worker
    pool = multiprocessing.pool.ThreadPool(processes=1)
    try:
        # Submit the task and get an async result
        async_result = pool.apply_async(controller.act, (obs,))
        # Wait for the result, with timeout
        return async_result.get(timeout=time_limit)
    except multiprocessing.TimeoutError:
        pool.terminate()  # Force-terminate the pool to stop the worker
        raise TimeoutError(f"Controller.act() took longer than {time_limit} seconds")
    finally:
        pool.close()

def play_controller_in_task(
    controller: Controller,
    task: Task,
    n_episodes: int,
    is_eval: bool,
    log_dir: str,
) -> FeedbackAggregated:
    """Play the controller in the task for n_episodes episodes and return the feedback.
    
    Args:
        controller (Controller): The controller to play in the task.
        task (Task): The task to play in.
        n_episodes (int): The number of episodes to play.
        is_eval (bool): Whether to evaluate the controller or not.
        log_dir (str): The subdirectory to log the results in. Results will be logged
            in each of <log_dir_global>/<log_dir>/<name_file>.txt
    """
    # Initialize the feedback
    feedback_agg_over_episodes = FeedbackAggregated()
    for k in range(n_episodes):
        # Reset the environment
        obs, info = task.reset(
            is_eval=is_eval and k == 0, log_dir=log_dir
        )  # eval only once per rollout for now
        task.render()

        # Loop over the episode
        done = False
        truncated = False
        error_message = None
        while not done and not truncated:
            # Act in the environment
            try:
                action = act_time_bounded(controller, obs, time_limit=4.0)
            except Exception as e: # Deal with error happening in the act method
                full_error_info = get_error_info(e)
                error_message = f"An error occured during the act method of the controller. Full error info : {full_error_info}"
                obs, reward, done, truncated = None, 0, False, True
                break
            # Step in the environment
            obs, reward, done, truncated, info = task.step(action)
            if "Error" in info: # Deal with error happening in the step method and logged in the info dict
                error_message = info["Error"]
                print(f"ERROR WARNING : {error_message}")
                break
            # Render and log
            task.render()

        # Env feedback
        env_feedback = task.get_feedback()

        # Close the environment
        task.close()

        # Create the feedback
        feedback = {
            "success": reward > 0,
            "reward": reward,
        }
        feedback.update(env_feedback)  # add environment feedback to feedback
        if error_message is not None:
            feedback["error"] = ErrorTrace(error_message)
            
        # Add feedback to the feedback aggregator
        feedback_agg_over_episodes.add_feedback(feedback)

    return feedback_agg_over_episodes