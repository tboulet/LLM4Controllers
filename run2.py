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
from core.play import play_controller_in_task
from env import env_name_to_MetaEnvClass
from agent import agent_name_to_AgentClass, agent2_name_to_AgentClass

register_hydra_resolvers()


@hydra.main(
    config_path="configs", config_name="config_default2.yaml", version_base="1.3.2"
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

    log_dir: str = config["log_dir"]
    do_cli: bool = config["do_cli"]
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
    run_name = f"{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_{agent_name}_{env_name}_seed{seed}"
    config["agent"]["config"]["run_name"] = run_name
    config["env"]["config"]["run_name"] = run_name
    config["llm"]["run_name"] = run_name

    # Initialize loggers
    shutil.rmtree(
        f"{log_dir}/last/", ignore_errors=True
    )  # Remove log_dir/last/ to clean the logs
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

    # Create the env
    print("Creating the dataset...")
    MetaEnvClass = env_name_to_MetaEnvClass[env_name]
    env = MetaEnvClass(config["env"]["config"], logger=logger)

    # Give LLM config to the agent
    config["agent"]["config"]["llm"] = config["llm"]

    # Create the agent
    print("Creating the agent...")
    AgentClass = agent2_name_to_AgentClass[agent_name]
    agent = AgentClass(config=config["agent"]["config"], logger=logger, env=env)

    # Training loop
    step = 0
    while step < n_steps_max:
        agent.step()
        step += 1


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
