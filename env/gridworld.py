import enum
import random
from matplotlib import pyplot as plt
import numpy as np
from env.base_meta_env import BaseMetaEnv, Observation, InfoDict
from typing import Tuple, Union, Dict, Any, List, Optional


class ObservationGridworld(Observation):
    def __init__(self, map: np.ndarray, position: Tuple[int, int]) -> None:
        self.map = map
        self.position = position


dict_actions = {
    "UP": "Move the agent up",
    "DOWN": "Move the agent down",
    "LEFT": "Move the agent left",
    "RIGHT": "Move the agent right",
}


class GridworldMetaEnv(BaseMetaEnv):

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        # Initialize the grid environment
        self.height, self.width = config["dimensions"]
        self.max_steps = config.get("max_steps", 30)
        self.tasks: List[str] = config["tasks"]
        # Initialize rendering
        self.rendering_active = False

    def get_textual_description(self):
        action_desc_listing = "\n".join(
            [
                f"- {action}: {description}"
                for action, description in dict_actions.items()
            ]
        )
        return f"""The Gridworld environment is a 2D grid in which the agent can move in 4 directions (up, down, left, right).

Each episode, the agent will be faced with a certain taks among a variety of tasks.
These can include navigation tasks (move to a certain location), logical tasks (find the nearest point among a list), manipulative tasks (build a wall), etc.

Actions: The action space consist of the following actions:\n{action_desc_listing}
Only those (strings) actions are allowed and should be taken by the controller.

Observations: The observation is an object composed of the following fields:
- "position" : a tuple of 2 integers representing the position of the agent on the map.
- other fields can be added depending on the task. This will be specified in the task description.
You can access those fields by calling the corresponding attribute of the observation object (e.g. obs.map, obs.position).
"""

    # TODO : add the following map to the description
    # - "map" : a 3D numpy array representing the map of the environment of shape (height, width, n_channels).
    # The channels corresponds to the position of whether there is the agent (channel 0), and the wall obstacles (channel 1).

    def reset(
        self, seed: Union[int, None] = None
    ) -> Tuple[Observation, str, Dict[str, Any]]:
        # Initialize the grid environment
        self.grid = np.zeros((self.height, self.width, 2))  # Initialize grid
        self.agent_pos = random.choice([0, self.height - 1]), random.choice(
            [0, self.width - 1]
        )
        self.grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        self.t = 0
        self.obs = ObservationGridworld(self.grid, self.agent_pos)
        # Initialize the task
        self.name_task = random.choice(self.tasks)
        if self.name_task == "go_to_center":
            task_description = f"Go to the center of the grid, i.e. at coordinates ({self.height // 2}, {self.width // 2})."
        # Initialize rendering
        self.block_render_this_episode = False
        # Return reset feedback
        return self.obs, task_description, {"task": self.name_task}

    def step(self, action: str) -> Tuple[Observation, float, bool, InfoDict]:
        # Update agent position
        self.grid[self.agent_pos[0], self.agent_pos[1], 0] = 0
        if action == "UP":
            self.agent_pos = max(0, self.agent_pos[0] - 1), self.agent_pos[1]
        elif action == "DOWN":
            self.agent_pos = (
                min(self.height - 1, self.agent_pos[0] + 1),
                self.agent_pos[1],
            )
        elif action == "LEFT":
            self.agent_pos = self.agent_pos[0], max(0, self.agent_pos[1] - 1)
        elif action == "RIGHT":
            self.agent_pos = self.agent_pos[0], min(
                self.width - 1, self.agent_pos[1] + 1
            )
        else:
            raise ValueError(f"Invalid action {action} (type: {type(action)}).")
        self.grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        # Check if the task is solved
        if self.name_task == "go_to_center":
            done = self.agent_pos == (self.height // 2, self.width // 2)
            reward = 1 if done else 0
        else:
            raise ValueError(f"Invalid task {self.name_task}.")
        # Advance step counter and check termination
        self.t += 1
        truncated = (self.t >= self.max_steps)
        # Return step feedback
        self.obs = ObservationGridworld(self.grid, self.agent_pos)
        return self.obs, reward, done, truncated, {}

    def render(self):
        # Don't render if rendering is blocked
        if self.block_render_this_episode:
            return

        # Define a function to close the rendering
        def on_close(event):
            plt.close(self.fig)
            self.block_render_this_episode = True

        # Initialize rendering if not active
        if not self.rendering_active:
            self.rendering_active = True
            self.fig, self.ax = plt.subplots()
            self.fig.canvas.mpl_connect("close_event", on_close)
            plt.ion()

        # Clear previous frame
        self.ax.clear()

        # Create an RGB grid representation
        grid_display = np.ones((self.height, self.width, 3))  # White background

        # Define colors
        agent_color = [0, 0, 1]  # Blue for agent
        wall_color = [0, 0, 0]  # Black for walls

        # Apply colors
        grid_display[self.grid[:, :, 0] == 1] = agent_color  # Agent
        grid_display[self.grid[:, :, 1] == 1] = wall_color  # Walls

        # Display grid
        self.ax.imshow(grid_display, origin="upper")
        self.ax.set_xticks(range(self.width))
        self.ax.set_yticks(range(self.height))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        plt.draw()
        plt.pause(0.2)

    def close(self):
        if self.rendering_active:
            plt.close(self.fig)
            self.rendering_active = False
