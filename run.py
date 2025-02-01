# Logging
import os
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
    n_episodes: int = config["n_iterations"]
    do_cli: bool = config["do_cli"]
    do_wandb: bool = config["do_wandb"]
    do_tb: bool = config["do_tb"]
    do_tqdm: bool = config["do_tqdm"]

    # Set the seeds
    seed = try_get_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

    # Create the env
    print("Creating the dataset...")
    EnvClass = env_name_to_MetaEnvClass[env_name]
    env = EnvClass(config["env"]["config"])
    textual_description_env = env.get_textual_description()
    print(textual_description_env)
    
    # Get the agent
    print("Creating the agent...")
    AgentClass = agent_name_to_AgentClass[agent_name]
    agent = AgentClass(config=config["agent"]["config"])
    if True: # Should only apply to agents that are supposed to receive the textual description of the environment
        agent.give_textual_description(textual_description_env)

    # Initialize loggers
    run_name = f"[{agent_name}]_[{env_name}]_{datetime.datetime.now().strftime('%dth%mmo_%Hh%Mmin%Ss')}_seed{np.random.randint(seed)}"
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

    # Training loop
    for ep in tqdm(range(n_episodes), disable=not do_tqdm):
        
        # Reset the environment
        obs, task, info = env.reset(seed=seed)
        if len(info) > 0:
            print(f"Episode {ep} - Info: {info}")
            
        # Ask the agent to generate a controller for the task
        controller = agent.get_controller(task)
        
        # Loop over the episode
        done = False
        ep_reward = 0
        while not done:
            # Act in the environment
            action = controller.act(obs)
            # Step in the environment
            obs, reward, done, info = env.step(action)
            # Render and log
            env.render()
            ep_reward += reward

        # Close the environment
        env.close()
        
        # Log the episode
        if do_tb:
            tb_writer.add_scalar("reward", ep_reward, ep)
        if do_cli:
            print(f"Episode {ep} - Reward: {ep_reward}")
        pass # pass other logger for now
    
        
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
