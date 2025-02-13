import enum
import random
import re
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple, Type, Union, Dict, Any, List, Optional

import minigrid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs import EmptyEnv, GoToObjectEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper, FullyObsWrapper

from env.base_meta_env import BaseMetaEnv, Observation, InfoDict
from core.task import TaskRepresentation
from core.spaces import FiniteSpace
from core.curriculums import CurriculumByLevels
from core.types import ActionType


from .env_minigrid_autosuccess import AutoSuccessMGEnv
from .env_minigrid_return_agent_position import GiveAgentPositionMGEnv

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
        self.size = config.get("size", 4)
        self.render_mode = config.get("render_mode", None)
        self.render_mode_eval = config.get("render_mode_eval", None)
        # Define the default observation and action spaces
        self.observation_space_default = spaces.Dict(
            direction=spaces.Discrete(4),
            image=spaces.Box(
                low=0,
                high=255,
                # shape=(self.viewsize, self.viewsize, 3),
                shape=(self.size, self.size, 3),
                dtype="uint8",
            ),
        )
        self.action_space_default = FiniteSpace(
            elems=sorted(list(dict_actions.keys()), key=lambda x: dict_actions[x][0])
        )
        # Define the curriculum
        self.family_tasks_to_env_class = {
            "do nothing particular": AutoSuccessMGEnv,
            "Give the position of the agent as a tuple of integers (x, y)": GiveAgentPositionMGEnv,
            "go to the <color> <obj_type>": GoToObjectEnv,
        }
        self.curriculum = CurriculumByLevels(
            levels=[
                {"do nothing particular"},
                {"Give the position of the agent as a tuple of integers (x, y)"},
                {"go to the <color> <obj_type>"},
            ]
        )
        # Call the parent class constructor
        super().__init__(config)

    # ======= Mandatory interface methods =======

    def get_textual_description(self) -> str:
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
    ) -> Tuple[Observation, TaskRepresentation, Dict[str, Any]]:
        # Select a family task
        family_task: str = self.curriculum.sample()
        ### === Create the environment === ###
        # Extract the environment class from the family task and instantiate it
        EnvClass: Type[MiniGridEnv] = self.family_tasks_to_env_class[family_task]
        self.env = EnvClass(
            size=self.size,
            render_mode=self.render_mode_eval if is_eval else self.render_mode,
        )
        # Wrap the environment to get the observation (for now we set the obs to be fully observable)
        self.env = FullyObsWrapper(self.env)
        # Modify the action space
        if hasattr(self.env.unwrapped, "get_new_action_space"):
            self.env.action_space = self.env.unwrapped.get_new_action_space()
        else:
            self.env.action_space = self.action_space_default
        # Reset the environment
        obs, info = self.env.reset()
        # Extract the task_representation from the environment
        self.task = self.extract_task(self.env)
        assert (
            self.task.family_task == family_task
        ), f"Family task mismatch: {self.task.family_task} != {family_task}"

        # assert isinstance(
        #     self.env.action_space, spaces.Discrete
        # ) and self.env.action_space.n == len(
        #     self.action_space.elems
        # ), f"Action space mismatch : {self.env.action_space} incompatible with {self.action_space}"
        # assert all(
        #     [
        #         key == "mission"
        #         or (self.observation_space[key] == self.env.observation_space[key])
        #         for key in self.observation_space.spaces.keys()
        #     ]
        # ), f"Observation space mismatch : {self.env.observation_space} incompatible with {self.observation_space}"

        # Get observation
        # del obs["mission"]
        # Return reset feedback
        info = {"task": self.task, **info}
        return obs, self.task, info

    def step(self, action: ActionType) -> Tuple[Observation, float, bool, InfoDict]:
        try:
            # Convert the action (e.g. "forward") to the action index (e.g. 2 (int))
            if not hasattr(self.env.unwrapped, "get_new_action_space"): # if the action space is not modified, perform "forward" -> 2
                action = dict_actions[action][0]
            # Take the action in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Return step feedback
            return obs, reward, terminated, truncated, info
        except KeyError:
            assert not action in dict_actions, f"Action '{action}' should not be in dict_actions here"
            info = {
                "error": {
                    "type": "action_error",
                    "message": f"Action '{action}' of type {type(action)} given to the .step() method of the environment is not an admissible action. Admissible actions are: {list(dict_actions.keys())}",
                }
            }
            return None, 0, True, False, info

    def update(self, task: TaskRepresentation, feedback: Dict[str, Any]) -> None:
        self.curriculum.update(objective=task.family_task, feedback=feedback)

    # ======= Optional interface methods =======

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

    # ======= Helper methods =======

    def extract_task(self, env: MiniGridEnv) -> TaskRepresentation:
        # The name of the task is the mission string
        name = env.unwrapped.mission
        # The description of the task is the env's task description if any, else the mission string
        description = (
            env.unwrapped.task_description
            if hasattr(env.unwrapped, "task_description")
            else name
        )
        # The family task is the mission function signature
        mission_space: MissionSpace = env.observation_space["mission"]
        mission_func = (
            mission_space.mission_func
        )  # f : color, obj_type -> go to the {color} {obj_type}
        keys_kwargs = mission_func.__code__.co_varnames  # ['color', 'obj_type']
        family_task = mission_func(  # "go to the <color> <obj_type>"
            *[f"<{keys_kwargs[i]}>" for i in range(len(keys_kwargs))]
        )
        # Extract the values of the placeholders in the family_task
        kwargs = self.extract_values(family_task, name, keys_kwargs)
        # Create the task representation
        task = TaskRepresentation(
            name=name,  # go to the green ball
            family_task=family_task,  # go to the <color> <obj_type>
            description=description,  # go to the green ball using your navigation skills
            kwargs=kwargs,  # {"color": "green", "obj_type": "ball"}
            # observation_space=self.extract_observation_space_without_mission(
            #     env.observation_space
            # ),  # gym.space object
            observation_space=env.observation_space,  # gym.space object
            action_space=env.action_space,  # gym.space object
        )
        return task

    def extract_values(self, family_task, name, keys_kwargs):
        # Create a regex pattern based on the family string
        family_task_escape = re.escape(family_task)
        pattern = family_task_escape.replace("<", "(?P<").replace(">", ">[^ ]+)")
        # Use the regex pattern to match the name string
        match = re.match(pattern, name)
        if not match:
            breakpoint()
            raise ValueError(
                f"The name does not match the family pattern : family_task={family_task}, name={name}, pattern={pattern}"
            )
        # Extract the values and create the dictionary
        return {key: match.group(key) for key in keys_kwargs}

    def extract_observation_space_without_mission(
        self, observation_space: spaces.Dict
    ) -> spaces.Dict:
        return spaces.Dict(
            {k: v for k, v in observation_space.spaces.items() if k != "mission"}
        )

    def get_feedback(self):
        if hasattr(self.env, "get_feedback"):
            return self.env.unwrapped.get_feedback()
        else:
            return super().get_feedback()
