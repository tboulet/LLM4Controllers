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
from tbutils.config import try_get

# ML libraries
import random
import numpy as np
import torch
import transformers

# Project imports
from agent.base_controller import Controller
from core.types import ErrorTrace
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.cli import LoggerCLI
from core.loggers.csv import LoggerCSV
from core.loggers.multi_logger import MultiLogger
from core.loggers.tensorboard import LoggerTensorboard
from core.loggers.tqdm_logger import LoggerTQDM
from core.loggers.profiler import LoggerProfiler
from core.task import Task
from core.utils import get_error_info, sanitize_name
from core.register_hydra import register_hydra_resolvers
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
    model_name: str = try_get(config["agent"], "config.llm.model", default="")
    model_name = model_name.split("/")[-1]  # Get the last part of the model name (e.g., "gpt-3.5-turbo")
    model_name = sanitize_name(model_name)

    n_steps_max: int = config.get("n_steps_max", np.inf)
    n_steps_max = to_maybe_inf(n_steps_max)

    log_dir: str = config["log_dir"]
    do_cli: bool = config["do_cli"]
    do_tb: bool = config["do_tb"]
    do_csv: bool = config["do_csv"]
    do_tqdm: bool = config["do_tqdm"]
    do_profiling: bool = config["do_profiling"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    print(f"Using seed: {seed}")

    # Set the run name
    date = datetime.datetime.now().strftime("%mmo%dth_%Hh%Mmin%Ss")
    run_name = "_".join([date, agent_name, model_name, env_name, f"seed{seed}"])
    run_name = config.get(
        "run_name", run_name
    )  # do "python run.py [your args] +run_name=<your_run_name>" to change the run name
    config["agent"]["config"]["run_name"] = run_name
    config["env"]["config"]["run_name"] = run_name
    config["llm"]["run_name"] = run_name
    if "llm" in config["agent"]["config"]:
        config["agent"]["config"]["llm"]["run_name"] = run_name

    # Initialize loggers
    shutil.rmtree(
        f"{log_dir}/last/", ignore_errors=True
    )  # Remove log_dir/last/ to clean the logs
    print(f"\nStarting run {run_name} ...")
    loggers = []
    if do_cli:
        loggers.append(LoggerCLI())
    if do_tb:
        loggers.append(LoggerTensorboard(log_dir=f"{log_dir}/{run_name}"))
    if do_csv:
        loggers.append(
            LoggerCSV(
                log_dirs=[f"{log_dir}/{run_name}", f"{log_dir}/last/"],
                timestep_key="_step",
            )
        )
    if do_tqdm and n_steps_max != np.inf:
        loggers.append(LoggerTQDM(n_total=n_steps_max))
    if do_profiling:
        loggers.append(LoggerProfiler(log_dirs = [f"{log_dir}/{run_name}", log_dir]))
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
    while step < n_steps_max and not agent.is_done():
        agent.step()
        step += 1

    # Close loggers
    logger.close()

if __name__ == "__main__":
    main()
