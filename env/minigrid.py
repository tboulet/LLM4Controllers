import enum
import random
import re
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Union, Dict, Any, List, Optional

import minigrid
from minigrid.envs import EmptyEnv, GoToObjectEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper, FullyObsWrapper
from env.base_meta_env import BaseMetaEnv, Observation, InfoDict
from core.task import TaskRepresentation
from core.spaces import FiniteSpace


dict_actions = {
    "left": (0, "Turn the direction of the agent to the left"),
    "right": (1, "Turn the direction of the agent to the right"),
    "forward": (2, "Move one tile forward"),
    "pickup": (
        3,
        "Pick up the object the agent is facing (if any) and add it to the agent's inventory",
    ),
    "drop": (
        4,
        "Drop the object from the agent's inventory (if any) in front of the agent",
    ),
    "toggle": (5, "Toggle/activate an object in front of the agent"),
    "done": (6, "Done completing the task"),
}


class MinigridMetaEnv(BaseMetaEnv):

    def __init__(self, config: Dict) -> None:
        # Extract parameters from the configuration
        self.viewsize = config.get("viewsize", 7)
        self.size = config.get("size", 10)
        self.render_mode = config.get("render_mode", None)
        self.render_mode_eval = config.get("render_mode_eval", None)
        # Define the default observation and action spaces
        self.observation_space = spaces.Dict(
            direction=spaces.Discrete(4),
            image=spaces.Box(
                low=0,
                high=255,
                # shape=(self.viewsize, self.viewsize, 3),
                shape=(self.size, self.size, 3),
                dtype="uint8",
            ),
        )
        self.action_space = FiniteSpace(
            elems=sorted(list(dict_actions.keys()), key=lambda x: dict_actions[x][0])
        )
        # Call the parent class constructor
        super().__init__(config)

    def get_textual_description(self):
        action_desc_listing = "\n".join(
            [
                f"- {action}: {idx_and_desc[1]}"
                for action, idx_and_desc in dict_actions.items()
            ]
        )
        return f"""The environment is a collection of 2D gridworld-like tasks where the agent can move forward, turn left or right and interact with objects (pick up, drop, toggle) in the environment.
These tasks have in common an agent with a discrete action space that has to navigate a 2D map with different obstacles (Walls, Lava, Dynamic obstacles) depending on the task. 
The task to be accomplished is described by a mission string (such as "go to the green ball", "open the door with the red key", etc.).
These mission tasks include different goal-oriented and hierarchical missions such as picking up boxes, opening doors with keys or navigating a maze to reach a goal location.
Each episode, the agent will be faced with a certain taks among a variety of tasks.
These can include navigation tasks (move to a certain location), logical tasks (find the nearest point among a list), manipulative tasks (build a wall), etc.

Actions: The action space consist of the following actions:\n{action_desc_listing}
Only those (str objects) actions are allowed and should be taken by the controller.

Observations: The observation is a dictionary with the following keys:
- direction: the direction the agent is facing (0: up, 1: right, 2: down, 3: left)
- image: the agent's view of the environment as a 3D numpy array of shape (viewsize, viewsize, 3). The channels represent the encoding of the object at position (i,j) in the environment (object type, color, state).
"""

    def reset(
        self,
        seed: Union[int, None] = None,
        is_eval: bool = False,
    ) -> Tuple[Observation, str, Dict[str, Any]]:
        # Select a task
        EnvClass = EmptyEnv
        # EnvClass = GoToObjectEnv
        self.env = EnvClass(
            size=self.size,
            render_mode=self.render_mode_eval if is_eval else self.render_mode,
        )  # TODO : change this to task selection process
        self.env = FullyObsWrapper(self.env)
        obs, info = self.env.reset()
        self.task: str = self.env.env.mission
        assert isinstance(
            self.env.action_space, spaces.Discrete
        ) and self.env.action_space.n == len(
            self.action_space.elems
        ), f"Action space mismatch : {self.env.action_space} incompatible with {self.action_space}"
        assert all(
            [
                key == "mission"
                or (self.observation_space[key] == self.env.observation_space[key])
                for key in self.observation_space.spaces.keys()
            ]
        ), f"Observation space mismatch : {self.env.observation_space} incompatible with {self.observation_space}"
        # Get observation
        del obs["mission"]
        # Return reset feedback
        info = {"task": self.task, **info}
        return obs, self.task, info

    def step(self, action: str) -> Tuple[Observation, float, bool, InfoDict]:
        try:
            # Convert the action (e.g. "forward") to the action index (e.g. 2 (int))
            action_idx = dict_actions[action][0]
            # Take the action in the environment
            obs, reward, terminated, truncated, info = self.env.step(action_idx)
            # Get observation
            del obs["mission"]
            # Return step feedback
            return obs, reward, terminated, truncated, info
        except Exception as e:
            if not action in dict_actions:
                info = {
                    "error": {
                        "type": "action_error",
                        "message": f"Action '{action}' of type {type(action)} given to the .step() method of the environment is not an admissible action. Admissible actions are: {list(dict_actions.keys())}",
                    }
                }
                return None, 0, True, False, info
            else:
                assert (
                    "mission" in obs
                ), f"Observation should contain the mission string"
                raise f"An error occured during the step method of the environment: {e}"

    def render(self):
        if self.render_mode == "human":
            self.env.render()
        elif self.render_mode == "rgb_array":
            return self.env.render(
                "rgb_array"
            )  # TODO : implement logging of the rgb_array
        elif self.render_mode == None:
            pass
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def close(self):
        self.env.close()

    def extract_task(self):
        mission_space: MissionSpace = self.env.observation_space["mission"]
        mission_func = (
            mission_space.mission_func
        )  # f : color, obj_type -> go to the {color} {obj_type}
        names_variables = mission_func.__code__.co_varnames  # ['color', 'obj_type']
        name = mission_func(  # "go to the {color} {obj_type}"
            *[f"<{names_variables[i]}>" for i in range(len(names_variables))]
        )
        task = TaskRepresentation(
            name=name,
            description=self.env.mission,
            variables=names_variables,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        raise NotImplementedError
        return task

    def get_observation_space(self) -> spaces.Space:
        return self.observation_space

    def get_action_space(self) -> spaces.Space:
        return self.action_space
