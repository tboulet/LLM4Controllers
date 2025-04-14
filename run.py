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

# Project imports
from agent.base_controller import Controller
from core.error_trace import ErrorTrace
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.cli import LoggerCLI
from core.loggers.multi_logger import MultiLogger
from core.loggers.tensorboard import LoggerTensorboard
from core.loggers.tqdm_logger import LoggerTQDM
from core.task import Task
from core.utils import get_error_info
from core.register_hydra import register_hydra_resolvers
from core.time_measure import RuntimeMeter
from core.utils import try_get_seed, to_maybe_inf
from env import env_name_to_MetaEnvClass
from agent import agent_name_to_AgentClass

register_hydra_resolvers()


@hydra.main(
    config_path="configs", config_name="config_default.yaml", version_base="1.3.2"
)
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

    # Get the config values from the config object.
    agent_name: str = config["agent"]["name"]
    env_name: str = config["env"]["name"]

    n_steps_max: int = config.get("n_steps_max", np.inf)
    n_steps_max = to_maybe_inf(n_steps_max)
    n_training_steps: int = config["n_training_steps"]
    n_eval_steps: int = config["n_eval_steps"]
    n_episodes_per_step: int = config["n_episodes_per_step"]

    log_dir: str = config["log_dir"]
    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    print(f"Using seed: {seed}")

    # Set the run name
    run_name = f"{agent_name}_{env_name}_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    config["agent"]["config"]["run_name"] = run_name
    config["env"]["config"]["run_name"] = run_name
    config["llm"]["run_name"] = run_name

    # Create the env
    print("Creating the dataset...")
    MetaEnvClass = env_name_to_MetaEnvClass[env_name]
    env = MetaEnvClass(config["env"]["config"])
    textual_description_env = env.get_textual_description()

    # Give LLM config to the agent
    config["agent"]["config"]["llm"] = config["llm"]

    # Create the agent
    print("Creating the agent...")
    AgentClass = agent_name_to_AgentClass[agent_name]
    agent = AgentClass(config=config["agent"]["config"])
    agent.give_textual_description(textual_description_env)

    # Initialize loggers
    run_name = config.get(
        "run_name", run_name
    )  # do "python run.py +run_name=<your_run_name>" to change the run name
    print(f"\nStarting run {run_name} ...")
    loggers = []
    if do_cli:
        loggers.append(LoggerCLI())
    if do_tb:
        loggers.append(LoggerTensorboard(log_dir=f"{log_dir}/{run_name}"))
    if do_tqdm and n_steps_max != np.inf:
        loggers.append(LoggerTQDM(n_total=n_steps_max))
    logger = MultiLogger(*loggers)

    # Remove log_dir/last/ to clean the logs
    shutil.rmtree(f"{log_dir}/last/", ignore_errors=True)

    # Training loop
    step = 0
    step_training = 0
    while step_training < n_steps_max:
        # Define training/evaluation mode
        if step % (n_training_steps + n_eval_steps) < n_eval_steps:
            is_eval = True
        else:
            is_eval = False
        # Get the task and its description
        task = env.get_task()
        task_description = task.get_description()

        # Ask the agent to generate a controller for the task
        controller = agent.get_controller(task_description)

        # Play the controller in the task
        def play_controller_in_task(
            controller: Controller,
            task: Task,
            n_episodes: int,
            is_eval: bool,
        ) -> FeedbackAggregated:
            """Play the controller in the task for n_episodes episodes and return the feedback."""
            # Initialize the feedback
            feedback_agg = FeedbackAggregated()
            for k in range(n_episodes):
                # Reset the environment
                obs, info = task.reset(is_eval=is_eval and k == 0) # eval only once per rollout for now
                task.render()

                # Loop over the episode
                done = False
                truncated = False
                while not done and not truncated:
                    # Act in the environment
                    try:
                        action = controller.act(obs)
                    except Exception as e:
                        full_error_info = get_error_info(e)
                        error_message = f"An error occured during the act method of the controller. Full error info : {full_error_info}"
                        info = {"Error": ErrorTrace(error_message)}
                        obs, reward, done, truncated = None, 0, False, True
                        break
                    # Step in the environment
                    obs, reward, done, truncated, info = task.step(action)
                    if "Error" in info:
                        error_message = info["Error"]
                        info["Error"] = ErrorTrace(error_message)
                        print(f"ERROR WARNING : {error_message}")
                        break
                    # Render and log
                    task.render()

                # Env feedback
                env_feedback = task.get_feedback()

                # Close the environment
                task.close()

                # Update the agent
                feedback = {
                    "success": reward > 0,
                    "reward": reward,
                }

                feedback.update(env_feedback)  # add environment feedback to feedback

                # Add feedback to the feedback aggregator
                feedback_agg.add_feedback(feedback)

            return feedback_agg

        feedback_agg = play_controller_in_task(
            controller, task, n_episodes=n_episodes_per_step, is_eval=is_eval
        )
        feedback_agg.aggregate()

        # Update the agent and the MetaEnv
        agent.update(task, task_description, controller, feedback_agg)
        env.update(task, feedback_agg)

        # Log the metrics
        metrics = feedback_agg.get_metrics()
        metrics.update(feedback_agg.get_metrics(task=task))
        logger.log_scalars(metrics, step=step)

        # Update the progress bar
        step += 1
        if not is_eval:
            step_training += 1


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
