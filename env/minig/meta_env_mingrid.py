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

from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.gotoobject import GoToObjectEnv
from minigrid.envs.crossing import CrossingEnv
from .env_minigrid_autosuccess import AutoSuccessEnv
from .env_minigrid_give_agent_position import GiveAgentPositionEnv
from .env_minigrid_give_goal_position import GiveGoalPositionEnv

from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper, FullyObsWrapper
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from minigrid.core.world_object import Goal, Lava, Wall, Key, Door

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

from env.base_meta_env import BaseMetaEnv, Observation, InfoDict
from core.task import Task, TaskRepresentation
from core.spaces import FiniteSpace
from core.curriculums import CurriculumByLevels
from core.types import ActionType


dict_actions = {
    "left": (
        0,
        "Turn the direction of the agent to the left (don't move in that direction)",
    ),
    "right": (
        1,
        "Turn the direction of the agent to the right (don't move in that direction)",
    ),
    "forward": (2, "Move one tile forward in the direction the agent is facing"),
    "pickup": (
        3,
        "Pick up the object the agent is facing (if any) and add it to the agent's inventory",
    ),
    "drop": (
        4,
        "Drop the object from the agent's inventory (if any) in front of the agent",
    ),
    "toggle": (5, "Toggle/activate an object in front of the agent"),
    "done": (6, "No action"),
}


class TaskMinigrid(Task):
    """Task class for the Minigrid environment.
    This is used to represent a task that can be performed in the Minigrid environment.
    """

    def __init__(self, creator_env_mg_func: Callable[..., MiniGridEnv]) -> None:
        self.creator_env_mg_func = creator_env_mg_func

    def __repr__(self) -> str:
        try:
            # Get the source code of the lambda function
            source = inspect.getsource(self.creator_env_mg_func).strip()

            # Extract the class name using regex (looking for ClassName(...) inside the lambda)
            match = re.search(r"(\w+)\s*\(", source)
            if match:
                class_name = match.group(1)
            else:
                class_name = "<unknown>"

            # Extract key-value arguments if present
            kwargs_match = re.findall(r"(\w+)\s*=\s*([\w./]+)", source)
            kwargs_repr = ", ".join(
                f"{k}={v}" for k, v in kwargs_match if k != "kwargs"
            )  # Ignore **kwargs

            return f"TaskMinigrid({class_name}({kwargs_repr}))"
        except Exception:
            return "TaskMinigrid(<unknown>)"

    def __call__(self, *args, **kwargs):
        return self.creator_env_mg_func(*args, **kwargs)


class MinigridMetaEnv(BaseMetaEnv):

    def __init__(self, config: Dict) -> None:
        # Extract parameters from the configuration
        self.viewsize = config.get("viewsize", 7)
        self.size = config.get("size", 10)
        self.render_mode = config.get("render_mode", None)
        self.render_mode_eval = config.get("render_mode_eval", None)
        # Define other parameters
        self.t = 0
        # Define the curriculum
        levels = [
            {
                lambda **kwargs: AutoSuccessEnv(size=self.size, **kwargs),
                lambda **kwargs: GiveAgentPositionEnv(size=self.size, **kwargs),
            },
            {
                lambda **kwargs: GiveGoalPositionEnv(size=self.size, **kwargs),
                lambda **kwargs: EmptyEnv(size=self.size, **kwargs),
            },
            {
                lambda **kwargs: EmptyEnv(
                    size=self.size,
                    agent_start_pos=np.random.randint(1, self.size // 2, size=2),
                    agent_start_dir=np.random.randint(4),
                    **kwargs,
                ),
            },
            {
                lambda **kwargs: GoToObjectEnv(size=self.size, **kwargs),
                lambda **kwargs: CrossingEnv(size=11, obstacle_type=Lava, **kwargs),
            },
        ]
        levels = [
            {TaskMinigrid(creator_env_mg_func) for creator_env_mg_func in level}
            for level in levels
        ]
        self.curriculum : CurriculumByLevels[TaskMinigrid] = CurriculumByLevels(levels)

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
For classical tasks, these string action should be returned by the act method of the agent.
However, some tasks will ask you to return an action of a different type (e.g. an integer, a tuple, etc.). In this case, you may be noticed by the task description.
In any case, the action space (and observation space) is given to you during the task description.

Observations: The observation is a dictionary with the following keys:
- direction: the direction the agent is facing (0: up, 1: right, 2: down, 3: left)
- image: the map of the environment as a 3D numpy array of shape (viewsize, viewsize, 3). The channels represent the encoding of the object at position (i,j) in the environment (object type, color, state). The environment is fully observable and the camera position and orientation are fixed (centered on the environment and facing up).
- mission: the mission string describing the task to be accomplished (e.g. "go to the green ball"). This should be the same as the task you will receive later so don't pay attention to it.

The mapping from object type integer to object type string is as follows: {IDX_TO_OBJECT}.
The mapping from color integer to color string is as follows: {IDX_TO_COLOR}.
The mapping from state integer to state string is as follows: {IDX_TO_STATE}. Only doors have a non-zero state.
For example, obs["image"][i,j] = [5, 2, 0] means that the object at position (i,j) is a key (object type 5) of color blue (color 2) in the open state (state 0).
"""

    def reset(
        self,
        seed: Union[int, None] = None,
        is_eval: bool = False,
    ) -> Tuple[Observation, Task, TaskRepresentation, Dict[str, Any]]:
        # Rendering and logging
        self.render_mode = self.render_mode_eval if is_eval else self.render_mode
        if self.render_mode == "rgb_array":
            self.video_frames: List[np.ndarray] = []
        config_logs = self.config["config_logs"]
        log_dir = config_logs["log_dir"]
        self.list_run_names = []
        if config_logs["do_log_on_new"]:
            self.list_run_names.append(self.config["run_name"])
        if config_logs["do_log_on_last"]:
            self.list_run_names.append("_last")
        # Select a task
        task = self.curriculum.sample()
        # Instancialize the minigrid environment
        self.env_mg = task(
            render_mode=self.render_mode,
        )
        # Wrap the environment to get the observation (for now we set the obs to be fully observable)
        self.env_mg = FullyObsWrapper(self.env_mg)
        # Modify the action space
        if hasattr(self.env_mg.unwrapped, "get_new_action_space"):
            self.env_mg.action_space = self.env_mg.unwrapped.get_new_action_space()
        else:
            self.env_mg.action_space = FiniteSpace(
                elems=sorted(
                    list(dict_actions.keys()), key=lambda x: dict_actions[x][0]
                )
            )
        # Reset the environment
        obs, info = self.env_mg.reset()
        # Extract the task_representation from the environment
        task_description = self.extract_task_repr_from_env_mg(self.env_mg)

        # Return reset elements
        info = {"task": task, **info}
        return obs, task, task_description, info

    def step(self, action: ActionType) -> Tuple[Observation, float, bool, InfoDict]:
        # Unsure the action is in the action space
        if not action in self.env_mg.action_space:
            return (
                None,
                0,
                True,
                False,
                {
                    "error": {
                        "type": "action_error",
                        "message": f"Action '{action}' of type {type(action)} is not in the env action space {self.env_mg.action_space}",
                    }
                },
            )
        # Convert the action (e.g. "forward") to the action index (e.g. 2 (int))
        if not hasattr(
            self.env_mg.unwrapped, "get_new_action_space"
        ):  # if the action space is not modified, perform "forward" -> 2
            action = dict_actions[action][0]
        # Take the action in the environment
        obs, reward, terminated, truncated, info = self.env_mg.step(action)
        # Return step feedback
        return obs, reward, terminated, truncated, info

    def update(self, task: Task, feedback: Dict[str, Any]) -> None:
        self.curriculum.update(objective=task, feedback=feedback)

    # ======= Optional interface methods =======

    def render(self):
        if self.render_mode == "human":
            pass  # already rendered in self.env.step/reset
        elif self.render_mode == "rgb_array":
            img = self.env_mg.render()
            self.video_frames.append(img)
        elif self.render_mode == None:
            pass
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def close(self):
        # Close the environment
        self.env_mg.close()
        # Save the video if any
        if self.render_mode == "rgb_array":
            config_logs = self.config["config_logs"]
            log_dir = config_logs["log_dir"]
            for run_name in self.list_run_names:
                path_task_t = os.path.join(log_dir, run_name, f"task_{self.t}")
                os.makedirs(path_task_t, exist_ok=True)
                path_task_t_video = os.path.join(path_task_t, "video.mp4")
                imageio.mimwrite(path_task_t_video, self.video_frames, fps=10)

        # Move forward time
        self.t += 1

    # ======= Helper methods =======

    def extract_task_repr_from_env_mg(self, env: MiniGridEnv) -> TaskRepresentation:
        """Extract the task representation from the minigrid environment of the task.

        Args:
            env (MiniGridEnv): the minigrid environment that will be used to extract the task description

        Returns:
            TaskRepresentation: the task representation extracted from the minigrid environment
        """
        # The name of the task is the mission string
        name = (
            f"Env: {env.unwrapped.__class__.__name__}\n"
            f"Mission: {env.unwrapped.mission}"
        )
        # The description of the task is the docstring of the env class
        description = env.unwrapped.__doc__
        # Create the task representation
        task = TaskRepresentation(
            name=name,  # go to the green ball
            description=description,  # go to the green ball using your navigation skills
            observation_space=env.observation_space,  # gym.space object
            action_space=env.action_space,  # gym.space object
        )
        return task

    def extract_kwargs(self, family_task, name, keys_kwargs):
        # Create a regex pattern based on the family string
        family_task_escape = re.escape(family_task)
        pattern = family_task_escape.replace("<", "(?P<").replace(">", ">[^ ]+)")
        # Use the regex pattern to match the name string
        match = re.match(pattern, name)
        if not match:
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
        if hasattr(self.env_mg.unwrapped, "get_feedback"):
            return self.env_mg.unwrapped.get_feedback()
        else:
            return super().get_feedback()
