from minigrid.minigrid_env import MiniGridEnv

# from gymnasium import spaces

# Python imports
import enum
import inspect
import os
import random
import re
import shutil
from gymnasium import spaces
import imageio
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Tuple, Type, Union, Dict, Any, List, Optional

# Minigrid imports
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper, FullyObsWrapper
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from minigrid.core.world_object import Goal, Lava, Wall, Key, Door

# Minig imports
from env.minig.utils import (
    IDX_TO_STATE,
    dict_actions,
    dict_directions,
)


class AAAMoveToPosition(MiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        # Optional things implemented here
        self.size = size
        self.possible_positions = [
            (x, y) for x in range(1, size - 1) for y in range(1, size - 1)
        ]
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.possible_positions],
        )

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(position: Tuple[int, int]):
        return f"go at position {position}"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(
            width, height
        )  # optional, recreate the grid (it is alreayd defined in __init__ of super class)

        # Generate the surrounding walls
        self.grid.wall_rect(
            0, 0, width, height
        )  # important to avoid falling in void I think

        # Place the agent
        self.agent_pos = np.random.randint(1, self.size - 2, size=2)
        self.agent_dir = np.random.randint(0, 4)

        # Select a random position and generate the mission
        self.position_target = self._rand_elem(self.possible_positions)
        self.mission = f"go at position {self.position_target}"

    def step(self, action):

        # MUST BE DONE (note : super().step() does all 4):
        # 1. update t and check if t >= max_steps
        # 2. deals with action
        # 3. render if render_mode is "human"
        # 4. generate observation
        # 5. return observation, reward, done, truncated, info

        # Run normal action
        obs, reward, done, truncated, info = super().step(action)

        # Check if the agent reached any of the direction
        x, y = self.agent_pos
        if x == self.position_target[0] and y == self.position_target[1]:
            done = True
            reward = self._reward()

        return obs, reward, done, truncated, info


def main():
    env = AAAMoveToPosition("up", render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
