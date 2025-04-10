# Python imports
import enum
import inspect
import os
import random
import re
import shutil
import textwrap
from gymnasium import spaces
import imageio
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Set, Tuple, Type, Union, Dict, Any, List, Optional

# Minigrid imports
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from minigrid.wrappers import ObservationWrapper, ImgObsWrapper, FullyObsWrapper
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from minigrid.core.world_object import Goal, Lava, Wall, Key, Door

# Env from minigrid
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.gotoobject import GoToObjectEnv
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.blockedunlockpickup import BlockedUnlockPickupEnv
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from minigrid.envs.fetch import FetchEnv
from minigrid.envs.fourrooms import FourRoomsEnv
from minigrid.envs.gotodoor import GoToDoorEnv
from minigrid.envs.gotoobject import GoToObjectEnv
from minigrid.envs.keycorridor import KeyCorridorEnv
from minigrid.envs.lavagap import LavaGapEnv
from minigrid.envs.distshift import DistShiftEnv
from minigrid.envs.lockedroom import LockedRoomEnv
from minigrid.envs.memory import MemoryEnv
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.envs.obstructedmaze import ObstructedMaze_1Dlhb, ObstructedMaze_Full
from minigrid.envs.obstructedmaze_v1 import ObstructedMaze_Full_V1
from minigrid.envs.playground import PlaygroundEnv
from minigrid.envs.putnear import PutNearEnv
from minigrid.envs.redbluedoors import RedBlueDoorEnv
from minigrid.envs.unlock import UnlockEnv
from minigrid.envs.unlockpickup import UnlockPickupEnv

# Env from BabyAI
from minigrid.envs.babyai.goto import GoToObj

# Env customs
from env.minig.env_minigrid_autosuccess import AutoSuccessEnv
from env.minig.env_minigrid_give_agent_position import GiveAgentPositionEnv
from env.minig.env_minigrid_give_goal_position import GiveGoalPositionEnv
from env.minig.env_minigrid_go_to_direction import GoTowardsDirection
from env.minig.env_minigrid_go_to_position import MoveToPosition

# Minig imports
from env.minig.utils import (
    IDX_TO_STATE,
    dict_actions,
    dict_directions,
)

# Project imports
from env.base_meta_env import BaseMetaEnv, Observation, InfoDict
from core.task import Task, TaskDescription
from core.spaces import FiniteSpace
from core.curriculums import CurriculumByLevels
from core.types import ActionType


def extract_sections(docstring: str, sections_docstring: list) -> Dict[str, str]:
    """
    Extract specified sections from a docstring formatted with Markdown-style headers (## Section Name).

    Args:
        docstring (str): The full docstring to parse.
        sections_to_extract (list): List of section titles to extract, e.g. ["Description", "Mission Space", "Termination"].

    Returns:
        Dict[str, str]: Dictionary mapping section names to their extracted text.
    """
    if docstring is None:
        return

    # Remove uniform indentation
    docstring = textwrap.dedent(docstring)

    # Match sections with headers like ## Description
    pattern = r"^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)"  # Matches section title and content
    matches = re.findall(pattern, docstring, re.DOTALL | re.MULTILINE)

    if matches == []:
        print("[WARNING] No section found in the docstring")

    list_sections = []
    for title, content in matches:
        title_clean = title.strip()
        if title_clean in sections_docstring:
            list_sections.append(f"## {title_clean} \n{content.strip()}\n")
    # Join the sections with new lines
    return "\n".join(list_sections)


class TaskMinigrid(Task):
    """Task class for the Minigrid environment.

    A task is specific to a goal : two instances of the same task must have the same goal.

    However some caracteristics (as the env size, the agent position, the goal position, etc.) can be different between two instances of the same task.

    This can be specified through the creator_env_mg_func :
    ```python
    creator_env_mg_func = lambda : SomeEnv(
        goal = <goal>,  # well defined goal
        agent_start_pos = np.random.randint(1, size - 2, size=2),  # varying-through-task agent start position
        )
    ```
    """

    def __init__(
        self,
        creator_env_mg_func: Callable[..., MiniGridEnv],
        sections_docstring: List[str],
        timestep: int,
        list_log_dirs_global: Optional[List[str]] = [],
        assert_task_identity_constancy: Optional[bool] = True,
        kwargs_env_mg: Optional[Dict[str, Any]] = {},
        render_mode_train: Optional[str] = None,
        render_mode_eval: Optional[str] = None,
    ) -> None:
        self.creator_env_mg_func = creator_env_mg_func
        self.sections_docstring = sections_docstring
        self.timestep = timestep
        self.list_log_dirs_global = list_log_dirs_global
        self.assert_task_identity_constancy = assert_task_identity_constancy
        self.kwargs_env_mg = kwargs_env_mg
        self.render_mode_train = render_mode_train
        self.render_mode_eval = render_mode_eval
        self.was_never_reset = True

        # Get the task name from the creator_env_mg_func
        self.func_str = inspect.getsource(self.creator_env_mg_func).strip()
        match = re.match(r"\s*lambda\s+[^\:]*\:\s*(.+)", self.func_str)
        assert (
            match is not None
        ), f"Could not extract the task name from the function string: {self.func_str}"
        self.func_str = match.group(1)

        # Instantiate the environment for the first time and do some wrapping
        env_mg = self.create_new_env_mg(render_mode=None)
        env_mg.reset()

        # Build the task description
        self.name = f"{self.func_str} - Mission : {env_mg.unwrapped.mission}"
        self.task_description = self.extract_task_description_from_env(env_mg)

        # Other variables
        self.n_times_timestep_logged = 0
        
    # ===== Helper methods ===

    def extract_task_description_from_env(self, env_mg: MiniGridEnv) -> TaskDescription:
        """Extract the task description from the environment.

        Args:
            env_mg (MiniGridEnv): The environment to extract the task description from.

        Returns:
            TaskDescription: The task description.
        """
        return TaskDescription(
            name=f"{self.func_str} - Mission : {env_mg.unwrapped.mission}",
            description=extract_sections(
                docstring=env_mg.unwrapped.__doc__,
                sections_docstring=self.sections_docstring,
            ),
            observation_space=env_mg.observation_space,
            action_space=env_mg.action_space,
        )

    def create_new_env_mg(self, **kwargs) -> MiniGridEnv:
        env_mg = self.creator_env_mg_func(**kwargs)
        env_mg = FullyObsWrapper(env_mg)
        if hasattr(env_mg, "get_new_action_space"):
            env_mg.action_space = env_mg.unwrapped.get_new_action_space()
        else:
            env_mg.action_space = FiniteSpace(
                elems=sorted(
                    list(dict_actions.keys()), key=lambda x: dict_actions[x][0]
                )
            )
        return env_mg

    # ===== Mandatory interface methods ======

    def get_description(self) -> TaskDescription:
        return self.task_description

    def reset(self, is_eval: str) -> Tuple[Observation, InfoDict]:
        """Reset the task to its initial state.

        Returns:
            Tuple[Observation, InfoDict]: The observation and info dictionary.
        """

        # Initialize rendering
        if is_eval:
            self.render_mode = self.render_mode_eval
        else:
            self.render_mode = self.render_mode_train

        if self.render_mode == "rgb_array":
            self.video_frames = []

        # Reset the environment
        self.env_mg = self.create_new_env_mg(render_mode=self.render_mode)
        obs, info = self.env_mg.reset()

        # Build info
        info = {"task_name": self.name, **info}

        # Check if the task description is the same as the first one
        if self.assert_task_identity_constancy:
            new_task_description = self.extract_task_description_from_env(self.env_mg)
            if new_task_description != self.task_description:
                raise ValueError(
                    f"Task identity constancy check failed: {self.task_description} != {new_task_description}"
                )

        # Return the observation and info
        return obs, info

    def step(
        self,
        action: ActionType,
    ) -> Tuple[
        Observation,
        float,
        bool,
        bool,
        InfoDict,
    ]:
        """Take a step in the task.

        Args:
            action (ActionType): The action to take.

        Returns:
            Tuple[Observation, float, bool, bool, InfoDict]: The observation, reward, done, truncated and info.
        """
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
        if not hasattr(self.env_mg.unwrapped, "get_new_action_space"):
            action = dict_actions[action][0]
        # Take the action in the environment
        obs, reward, terminated, truncated, info = self.env_mg.step(action)
        # Return step feedback
        info = {"task_name": self.name, **info}
        return obs, reward, terminated, truncated, info

    # ===== Other methods ======

    def render(self) -> None:
        if self.render_mode == "human":
            pass  # already rendered in self.env_mg.step/reset
        elif self.render_mode == "rgb_array":
            img = self.env_mg.render()
            self.video_frames.append(img)
        elif self.render_mode == None:
            pass
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def close(self) -> None:
        # Close the environment
        self.env_mg.close()
        # Save the video if any
        if self.render_mode == "rgb_array":
            self.n_times_timestep_logged += 1
            if self.n_times_timestep_logged > 1:
                dir_task = f"task_{self.timestep}_{self.n_times_timestep_logged}"
            else:
                dir_task = f"task_{self.timestep}"
            for log_dir_global in self.list_log_dirs_global:
                path_task_t = os.path.join(log_dir_global, dir_task)
                os.makedirs(path_task_t, exist_ok=True)
                path_task_t_video = os.path.join(path_task_t, "video.mp4")
                imageio.mimwrite(path_task_t_video, self.video_frames, fps=10)

    def get_feedback(self) -> Dict[str, Any]:
        if hasattr(self.env_mg.unwrapped, "get_feedback"):
            return self.env_mg.unwrapped.get_feedback()
        else:
            return {}

    def __repr__(self) -> str:
        return self.name


class MinigridMetaEnv(BaseMetaEnv):

    def __init__(self, config: Dict) -> None:
        # --- Extract parameters from the configuration file ---
        # Env parameters
        self.viewsize = config.get("viewsize", 7)
        self.size = config.get("size", 10)
        # Representation parameter
        self.sections_docstring = config.get("sections_docstring")
        # Logging and render
        self.log_dir = config["config_logs"]["log_dir"]
        self.render_mode_train = config.get("render_mode_train", None)
        self.render_mode_eval = config.get("render_mode_eval", None)

        # Define variables
        self.timestep = 0
        # Define the curriculum
        levels: List[Set[Callable[..., MiniGridEnv]]] = [
            # {
            #     # For testing envs
            #     lambda **kwargs: GoToObj(room_size=self.size, **kwargs),
            # },
            {
                # Observation structure comprehension and navigation comprehension tasks
                lambda **kwargs: GoTowardsDirection(
                    size=self.size, direction="up", **kwargs
                ),
                lambda **kwargs: MoveToPosition(
                    size=self.size, position=(3, 3), **kwargs
                ),
            },
            {
                # Navigation tasks (minimalistic)
                lambda **kwargs: EmptyEnv(size=self.size, **kwargs),
                lambda **kwargs: EmptyEnv(
                    size=self.size,
                    agent_start_pos=np.random.randint(1, self.size // 2, size=2),
                    agent_start_dir=np.random.randint(4),
                    **kwargs,
                ),
            },
            {
                # Navigation tasks (medium)
                lambda **kwargs: GoToDoorEnv(size=self.size, **kwargs),
                lambda **kwargs: CrossingEnv(size=11, obstacle_type=Wall, **kwargs),
                lambda **kwargs: GoToObjectEnv(size=self.size, numObjs=2, **kwargs),
                lambda **kwargs: FourRoomsEnv(**kwargs),
                lambda **kwargs: DistShiftEnv(**kwargs),
            },
            {
                # Navigation tasks (hard)
                lambda **kwargs: CrossingEnv(size=11, obstacle_type=Lava, **kwargs),
                lambda **kwargs: DynamicObstaclesEnv(
                    size=self.size, n_obstacles=4, **kwargs
                ),
                lambda **kwargs: LavaGapEnv(size=5, **kwargs),
                lambda **kwargs: MultiRoomEnv(minNumRooms=3, maxNumRooms=3, **kwargs),
            },
            {
                # Simple manipulative tasks (1 step)
                lambda **kwargs: FetchEnv(size=self.size, numObjs=3, **kwargs),
                lambda **kwargs: UnlockEnv(**kwargs),
            },
            {
                # Medium manipulative tasks (2-3 steps)
                lambda **kwargs: UnlockPickupEnv(**kwargs),
                lambda **kwargs: PutNearEnv(size=6, numObjs=2, **kwargs),
                lambda **kwargs: MemoryEnv(**kwargs),
                lambda **kwargs: DoorKeyEnv(**kwargs),
                lambda **kwargs: KeyCorridorEnv(**kwargs),
                lambda **kwargs: ObstructedMaze_1Dlhb(**kwargs),
                lambda **kwargs: RedBlueDoorEnv(**kwargs),
            },
            {
                # Hard manipulative tasks (4-5 steps or more)
                lambda **kwargs: BlockedUnlockPickupEnv(**kwargs),
                lambda **kwargs: PutNearEnv(size=14, numObjs=6, **kwargs),
                lambda **kwargs: LockedRoomEnv(size=19, **kwargs),
                lambda **kwargs: ObstructedMaze_Full(**kwargs),
                lambda **kwargs: ObstructedMaze_Full_V1(**kwargs),
            },
        ]
        self.curriculum: CurriculumByLevels[Callable[..., MiniGridEnv]] = (
            CurriculumByLevels(levels=levels)
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
For classical tasks, these string action should be returned by the act method of the agent.
However, some tasks will ask you to return an action of a different type (e.g. an integer, a tuple, etc.). In this case, you may be noticed by the task description.
In any case, the action space (and observation space) is given to you during the task description.

Observations: The observation is a dictionary with the following keys:
- direction (int) : the direction the agent is facing : {dict_directions}
- image (nd.array) : the map of the environment as a 3D numpy array of shape (viewsize, viewsize, 3). The channels represent the encoding of the object at position (i,j) in the environment (object type, color, state). The environment is fully observable and the camera position and orientation are fixed (centered on the environment and facing up).
- mission (str) : the mission string describing the task to be accomplished (e.g. "go to the green ball"). This should be the same as the task you will receive later so don't pay attention to it.

The mapping from object type integer to object type string is as follows: {IDX_TO_OBJECT}.
The mapping from color integer to color string is as follows: {IDX_TO_COLOR}.
The mapping from state integer to state string is as follows: {IDX_TO_STATE}. Only doors have a non-zero state.
For example, obs["image"][i,j] = [5, 2, 0] means that the object at position (i,j) is a key (object type 5) of color blue (color 2) in the open state (state 0).
"""

    def get_task(
        self,
    ) -> TaskMinigrid:
        # Rendering and logging
        config_logs = self.config["config_logs"]
        self.list_log_dirs_global = []
        if config_logs["do_log_on_new"]:
            self.list_log_dirs_global.append(
                os.path.join(self.log_dir, self.config["run_name"])
            )
        if config_logs["do_log_on_last"]:
            self.list_log_dirs_global.append(os.path.join(self.log_dir, "last"))
        # Select a task
        creator_env_mg_func = self.curriculum.sample()
        task = TaskMinigrid(
            creator_env_mg_func=creator_env_mg_func,
            sections_docstring=self.sections_docstring,
            list_log_dirs_global=self.list_log_dirs_global,
            assert_task_identity_constancy=True,  # true for now
            render_mode_train=self.render_mode_train,
            render_mode_eval=self.render_mode_eval,
            timestep=self.timestep,
        )
        # Instancialize the minigrid environment
        return task

    def update(self, task: TaskMinigrid, feedback: Dict[str, Any]) -> None:
        self.curriculum.update(objective=task, feedback=feedback)
        self.timestep += 1
