# Logging
import os
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
from typing import Dict, Type
import cProfile

# ML libraries
import random
import numpy as np

# Project imports
from core.utils import get_error_info
from src.time_measure import RuntimeMeter
from src.utils import try_get_seed
from env import env_name_to_MetaEnvClass
from agent import agent_name_to_AgentClass


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

    # Get the config values from the config object.
    agent_name: str = config["agent"]["name"]
    env_name: str = config["env"]["name"]
    n_episodes: int = config["n_episodes"]
    n_training_episodes: int = config["n_training_episodes"]
    n_eval_episodes: int = config["n_eval_episodes"]
    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Set the run name
    run_name = f"{agent_name}_{env_name}_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{seed}"
    config["agent"]["config"]["run_name"] = run_name
    config["env"]["config"]["run_name"] = run_name

    # Create the env
    print("Creating the dataset...")
    EnvClass = env_name_to_MetaEnvClass[env_name]
    env = EnvClass(config["env"]["config"])
    textual_description_env = env.get_textual_description()

    # Get the agent
    print("Creating the agent...")
    AgentClass = agent_name_to_AgentClass[agent_name]
    agent = AgentClass(config=config["agent"]["config"])
    agent.give_textual_description(textual_description_env)

    # Initialize loggers
    run_name = config.get(
        "run_name", run_name
    )  # do "python run.py +run_name=<your_run_name>" to change the run name
    os.makedirs("logs", exist_ok=True)
    print(f"\nStarting run {run_name}")
    if do_wandb:
        run = wandb.init(
            name=run_name,
            config=config,
            **config["wandb_config"],
        )
    if do_tb:
        tb_writer = SummaryWriter(log_dir=f"tensorboard/{run_name}")
    if do_tqdm:
        tqdm_bar = tqdm(
            total=n_episodes,
            disable=not do_tqdm and n_episodes != sys.maxsize,
        )
        tqdm_bar.set_description(f"Episode 0/{n_episodes}")

    # Training loop
    ep = 0
    ep_training = 0
    while ep_training < n_episodes:
        # Define training/evaluation mode
        if ep % (n_training_episodes + n_eval_episodes) < n_eval_episodes:
            is_eval = True
        else:
            is_eval = False
        # Reset the environment
        obs, task, info = env.reset(seed=seed, is_eval=is_eval)
        if len(info) > 0:
            print(f"Episode {ep} - Info: {info}")

        # Ask the agent to generate a controller for the task
        controller = agent.get_controller(task)

        # Loop over the episode
        done = False
        truncated = False
        while not done and not truncated:
            # Act in the environment
            try:
                action = controller.act(obs)
            except Exception as e:
                full_error_info = get_error_info(e)
                info = {
                    "error": f"An error occured during the act method of the controller.\n{full_error_info}"
                }
                obs, reward, done, truncated = (
                    None,
                    0,
                    True,
                    False,
                )
                print(f"WARNING : Error in the controller .act() : {info['error']}")
                break
            # Step in the environment
            obs, reward, done, truncated, info = env.step(action)
            if "error" in info:
                print(f"WARNING : Error in the environment .step() : {info['error']}")
                break
            # Render and log
            env.render()

        # Close the environment
        env.close()

        # Manage feedback
        feedback = {
            "success": reward > 0,
            "reward": reward,
        }
        if "error" in info:
            feedback["error"] = info["error"]
        agent.update(task, controller, feedback)

        # Log the episode
        print(f"Feedback for episode {ep} : {feedback}")

        # Update the progress bar
        if do_tqdm:
            tqdm_bar.update(1)
            tqdm_bar.set_description(f"Episode {ep}/{n_episodes}")
        if is_eval:
            ep += 1
        else:
            ep_training += 1
            ep += 1

    # Finish the WandB run.
    if do_wandb:
        run.finish()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("logs/profile_stats.prof")
    print("\nProfile stats dumped to profile_stats.prof")
    print(
        "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
    )
