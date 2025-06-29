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
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from core.utils import get_name_copy
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
from core.types import ActionType, TextualInformation


def extract_sections(docstring: str, list_sections_docstring: list) -> Dict[str, str]:
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
        if title_clean in list_sections_docstring:
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
        meta_env: "MinigridMetaEnv",
        assert_task_identity_constancy: Optional[bool] = True,
    ) -> None:
        self.creator_env_mg_func = creator_env_mg_func
        self.meta_env = meta_env
        self.assert_task_identity_constancy = assert_task_identity_constancy

        # Get the task name from the creator_env_mg_func
        self.func_str = inspect.getsource(self.creator_env_mg_func).strip()
        self.func_str = self.func_str.replace("lambda **kwargs: ", "")
        self.func_str = self.func_str.replace("**kwargs", "")
        self.func_str = re.sub(r"\s+", " ", self.func_str).strip()

        # Other variables
        self.task_description: str = None
        self.code_of_env_mg_class: str = None
        self.env_mg: MiniGridEnv = None
        self.first_obs: Observation = None

    # ===== Helper methods ===

    def extract_task_description_from_env(self, env_mg: MiniGridEnv) -> TaskDescription:
        """Extract the task description from the environment.

        Args:
            env_mg (MiniGridEnv): The environment to extract the task description from.

        Returns:
            TaskDescription: The task description.
        """
        return TaskDescription(
            # name=f"{self.func_str}",  # Mission : {env_mg.unwrapped.mission}
            description=extract_sections(
                docstring=env_mg.unwrapped.__doc__,
                list_sections_docstring=self.meta_env.list_sections_docstring,
            ),
            observation_space=env_mg.observation_space,
            action_space=env_mg.action_space,
        )

    def create_new_env_mg(self, **kwargs) -> MiniGridEnv:
        # Create the env based on the creator_env_mg_func
        env_mg = self.creator_env_mg_func(**kwargs)
        # Fully obs wrapper
        env_mg = FullyObsWrapper(env_mg)
        # Modify action space to FiniteSpace("forward", ...)
        if hasattr(env_mg.unwrapped, "get_new_action_space"):
            env_mg.action_space = env_mg.unwrapped.get_new_action_space()
        else:
            env_mg.action_space = FiniteSpace(
                elems=sorted(
                    list(dict_actions.keys()), key=lambda x: dict_actions[x][0]
                )
            )
        # Modify observation space's mission field so that it has nice repr
        mission_space = env_mg.observation_space["mission"]

        class MissionSpaceGoodRepr(mission_space.__class__):
            def __repr__(self: MissionSpace) -> str:
                args = inspect.signature(self.mission_func).parameters.keys()
                mission_sig = self.mission_func(**{arg: f"{{{arg}}}" for arg in args})
                if self.ordered_placeholders is not None:
                    ordered_placeholders = [
                        f"{order_pl[:10]}..." if len(order_pl) > 10 else order_pl
                        for order_pl in self.ordered_placeholders
                    ]
                    return f"MissionSpace(mission_template={mission_sig}, ordered_placeholders={ordered_placeholders})"
                else:
                    return f"MissionSpace(mission={mission_sig})"

        mission_space.__class__ = MissionSpaceGoodRepr
        return env_mg

    # ===== Mandatory interface methods ======

    def get_name(self) -> str:
        return self.func_str

    def get_description(self) -> TaskDescription:
        if self.task_description is None:
            if self.env_mg is None:
                self.env_mg = self.create_new_env_mg(render_mode=None)
            # Extract the task description from the environment
            self.task_description = self.extract_task_description_from_env(self.env_mg)
        return self.task_description

    def get_code_repr(self) -> str:
        if self.code_of_env_mg_class is None:
            if self.env_mg is None:
                self.env_mg = self.create_new_env_mg(render_mode=None)
            # Extract the code repr of the class of the environment
            self.code_of_env_mg_class = inspect.getsource(
                self.env_mg.unwrapped.__class__
            )
            docstring_of_class = self.env_mg.unwrapped.__class__.__doc__
            if docstring_of_class is not None:
                self.code_of_env_mg_class = self.code_of_env_mg_class.replace(
                    docstring_of_class, ""
                )
        return (
            "```python\n"
            f"{self.code_of_env_mg_class}\n"
            f"env = {self.func_str.replace(',', '')}\n"
            f"env = FullyObsWrapper(env)\n"
            f"env.action_space = FiniteSpace(\n"
            f"    elems={sorted(list(dict_actions.keys()), key=lambda x: dict_actions[x][0])}\n"
            "    )\n"
            "```"
        )

    def get_map_repr(self) -> str:
        """Get a string representation of the map of the environment.

        Returns:
            str: The string representation of the map.
        """
        if self.env_mg is None:
            self.env_mg = self.create_new_env_mg(render_mode=None)
        if self.first_obs is None:
            self.first_obs, _ = self.env_mg.reset()

        grid = self.first_obs["image"]

        # Step 1: Build grid of cell labels
        label_grid = []
        for row in grid:
            label_row = []
            for cell in row:
                obj_idx, color_idx, state_idx = cell
                obj = IDX_TO_OBJECT.get(obj_idx, "empty")
                color = IDX_TO_COLOR.get(color_idx, "")
                state = [k for k, v in STATE_TO_IDX.items() if v == state_idx]
                state_str = state[0] if state else None

                if obj == "empty":
                    label = ""
                elif obj == "wall":
                    label = "wall"
                elif obj == "door" and state_str:
                    label = f"{color} door ({state_str})"
                else:
                    label = f"{color} {obj}"
                label_row.append(label)
            label_grid.append(label_row)

        # Step 2: Find max label length
        max_len = max(len(label) for row in label_grid for label in row)
        num_cols = len(label_grid[0])
        hcell = "─" * (max_len + 2)

        # Step 3: Build box borders
        top = "┌" + "┬".join([hcell] * num_cols) + "┐"
        mid = "├" + "┼".join([hcell] * num_cols) + "┤"
        bottom = "└" + "┴".join([hcell] * num_cols) + "┘"

        # Step 4: Build rows with centered labels
        lines = [top]
        for i, row in enumerate(label_grid):
            centered_cells = [f" {label.center(max_len)} " for label in row]
            lines.append("│" + "│".join(centered_cells) + "│")
            if i < len(label_grid) - 1:
                lines.append(mid)
        lines.append(bottom)

        return "\n".join(lines)

    def reset(
        self, is_eval: str = False, log_dir: str = None
    ) -> Tuple[Observation, InfoDict]:
        """Reset the task to its initial state.

        Args:
            is_eval (bool): Whether to reset the environment in evaluation mode or not.
            log_dir (str): The directory to save the logs.

        Returns:
            Tuple[Observation, InfoDict]: The observation and info dictionary.
        """

        # Initialize rendering
        if is_eval:
            self.render_mode = self.meta_env.render_mode_eval
        else:
            self.render_mode = self.meta_env.render_mode_train
        self.log_dir = log_dir
        if self.render_mode == "rgb_array":
            self.video_frames = []

        # Reset the environment
        self.env_mg = self.create_new_env_mg(render_mode=self.render_mode)
        obs, info = self.env_mg.reset()
        self.first_obs = obs
        self.first_agent_pos = np.array(self.env_mg.unwrapped.agent_pos, dtype=np.int32)
        # Build info
        info = {"task_name": self.func_str, **info}

        # Check if the task description is the same as the first one
        if self.assert_task_identity_constancy:
            new_task_description = self.extract_task_description_from_env(self.env_mg)
            old_task_description = self.get_description()
            if new_task_description != old_task_description:
                raise ValueError(
                    f"Task identity constancy check failed: {old_task_description} != {new_task_description}"
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
        assert (
            self.env_mg is not None
        ), "The environment is not initialized. Please call reset() first."
        # Unsure the action is in the action space
        if not action in self.env_mg.action_space:
            return (
                None,
                0,
                True,
                False,
                {
                    "Error": f"Action '{action}' of type {type(action)} is not in the env action space {self.env_mg.action_space}",
                },
            )
        # Convert the action (e.g. "forward") to the action index (e.g. 2 (int))
        if not hasattr(self.env_mg.unwrapped, "get_new_action_space"):
            action = dict_actions[action][0]
        # Take the action in the environment
        obs, reward, terminated, truncated, info = self.env_mg.step(action)
        # Return step feedback
        return obs, reward, terminated, truncated, info

    # ===== Other methods ======

    def render(self) -> None:
        assert (
            self.env_mg is not None
        ), "The environment is not initialized. Please call reset() first."
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
        assert (
            self.env_mg is not None
        ), "The environment is not initialized. Please call reset() first."
        # Close the environment
        self.env_mg.close()
        del self.env_mg
        # Save the video if any
        if self.render_mode == "rgb_array":
            for log_dir_global in self.meta_env.list_log_dirs_global:
                path_task_t = os.path.join(log_dir_global, self.log_dir)
                os.makedirs(path_task_t, exist_ok=True)
                path_task_t_video = os.path.join(path_task_t, "video.mp4")
                while os.path.exists(path_task_t_video):
                    path_task_t_video = get_name_copy(path_task_t_video)
                imageio.mimwrite(path_task_t_video, self.video_frames, fps=10)

    def get_feedback(self) -> Dict[str, Any]:
        feedback_metrics = {}
        if "custom" in self.meta_env.list_feedback_keys and hasattr(
            self.env_mg.unwrapped, "get_feedback"
        ):
            feedback_metrics.update(self.env_mg.unwrapped.get_feedback())
        if "position" in self.meta_env.list_feedback_keys:
            feedback_metrics["final_agent_position"] = np.array(
                self.env_mg.unwrapped.agent_pos
            )
            feedback_metrics["first_agent_position"] = self.first_agent_pos
        if "duration" in self.meta_env.list_feedback_keys:
            feedback_metrics["episode_duration"] = self.env_mg.unwrapped.step_count
            if hasattr(self.env_mg.unwrapped, "max_steps"):
                feedback_metrics["episode_duration_normalized_by_max_duration"] = (
                    self.env_mg.unwrapped.step_count / self.env_mg.unwrapped.max_steps
                )
        if "distance_start_to_end" in self.meta_env.list_feedback_keys:
            feedback_metrics["distance_start_to_end"] = np.linalg.norm(
                np.array(self.env_mg.unwrapped.agent_pos) - self.first_agent_pos
            )
        if "map" in self.meta_env.list_feedback_keys:
            feedback_metrics["map_representation"] = TextualInformation(
                text=self.get_map_repr()
            )

        return feedback_metrics

    def __repr__(self) -> str:
        return self.get_name()


class MinigridMetaEnv(BaseMetaEnv):

    def __init__(self, config: Dict, logger: BaseLogger = None):
        self.config = config
        self.logger = logger
        # --- Extract parameters from the configuration file ---
        self.config = config
        # Env parameters
        self.viewsize = config.get("viewsize", 7)  # not used
        self.size = config.get("size", 10)
        # Feedback parameters
        self.list_feedback_keys = config.get("list_feedback_keys")
        # Representation parameter
        self.list_sections_docstring = config.get("list_sections_docstring")
        # Logging and render
        self.render_mode_train = config.get("render_mode_train", None)
        self.render_mode_eval = config.get("render_mode_eval", None)
        self.n_videos_logged = config.get("n_videos_logged", 1)
        config_logs = self.config["config_logs"]
        self.log_dir = config_logs["log_dir"]
        self.list_log_dirs_global = []
        if config_logs["do_log_on_new"]:
            self.list_log_dirs_global.append(
                os.path.join(self.log_dir, self.config["run_name"])
            )
        if config_logs["do_log_on_last"]:
            self.list_log_dirs_global.append(os.path.join(self.log_dir, "last"))

        # Define the curriculum
        levels = [
            # {
            #     # For testing envs
            #     lambda **kwargs: GoTowardsDirection(size=self.size, **kwargs),
            # },
            {
                # Observation structure comprehension and navigation comprehension tasks
                lambda **kwargs: GiveAgentPositionEnv(size=self.size, **kwargs),
                lambda **kwargs: GiveGoalPositionEnv(size=self.size, **kwargs),
                lambda **kwargs: GoTowardsDirection(size=self.size, **kwargs),
                lambda **kwargs: MoveToPosition(size=self.size, **kwargs),
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
                # lambda **kwargs: GoToObj(room_size=self.size, **kwargs), # not sure BabyAI envs are compatible
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
                lambda **kwargs: MemoryEnv(size=9, **kwargs),
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

        levels = [
            {
                TaskMinigrid(
                    creator_env_mg_func=creator_env_mg_func,
                    meta_env=self,
                )
                for creator_env_mg_func in level
            }
            for level in levels
        ]

        self.curriculum: CurriculumByLevels[Callable[..., MiniGridEnv]] = (
            CurriculumByLevels(levels=levels)
        )

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
- image (nd.array) : the full map of the environment as a 3D numpy array of shape (height, width, 3). The channels represent the encoding of the object at position (i,j) in the environment (object type, color, state). 
IMPORTANT : The environment is fully observable and the camera position and orientation are fixed (centered on the environment and facing up).
- mission (str) : the mission string describing the task to be accomplished (e.g. "go to the green ball").

The mapping from object type integer to object type string is as follows: {IDX_TO_OBJECT}.
The mapping from color integer to color string is as follows: {IDX_TO_COLOR}.
The mapping from state integer to state string is as follows: {IDX_TO_STATE}. Only doors have a non-zero state.
For example, obs["image"][i,j] = [5, 2, 0] means that the object at position (i,j) is a key (object type 5) of color blue (color 2) in the open state (state 0).
"""

    def get_env_usage_explanation_and_variables(self) -> Tuple[str, Dict[str, Any]]:
        tasks = self.get_current_tasks()
        tasks_repr = []
        for idx, task in enumerate(tasks):
            tasks_repr.append(f"- Task no {idx + 1} : {task}")
        tasks_repr = "\n".join(tasks_repr)
        explanation = f"""In this environment, you have access to several tasks to solve.
        The tasks are the following :
        {tasks_repr}
        
        The tasks are accessible through the variable `tasks`, which is a list of `TaskMinigrid` objects.
        To obtain task of index 3 for example, you can use ```task = tasks[3]```.
        
        There is certain methods usable to get details about the tasks:
        - `task.get_description()` : returns a `TaskDescription` object with the task description, observation space and action space that can be printed.
        - `task.get_code_repr()` : returns a string representation of the code of the task, including the environment class and the action space.
        - `task.get_map_repr()` : returns a string representation of the map of the environment, with the objects and their states.
        """
        return explanation, {"tasks" : tasks}

    def get_task(
        self,
    ) -> TaskMinigrid:
        return self.curriculum.sample()

    def get_current_tasks(self) -> List[TaskMinigrid]:
        """Get all the tasks currently available in the environment.

        Returns:
            List[TaskMinigrid]: the list of all tasks available in the environment
        """
        if self.config["do_curriculum"]:
            return self.curriculum.get_current_objectives()
        else:
            return [obj for level in self.curriculum.levels for obj in level.keys()]

    def update(self, task: TaskMinigrid, feedback: FeedbackAggregated) -> None:
        self.curriculum.update(objective=task, feedback=feedback)

    def get_code_repr(self) -> str:
        return (
            "```python\n"
            f"{inspect.getsource(MiniGridEnv)}\n"
            f"{inspect.getsource(FullyObsWrapper)}\n"
            "```"
        )
