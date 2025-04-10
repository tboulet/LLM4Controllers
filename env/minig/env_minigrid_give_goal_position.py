from typing import Optional
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

from core.error_trace import ErrorTrace


class GiveGoalPositionEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        # Optional things implemented here
        if max_steps is None:
            max_steps = 1
        self.failure_reason = None

        # MUST BE DONE : create mission space
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "give the position of the green goal as a tuple of integers (x, y)"

    def _gen_grid(self, width, height):
        # MUST BE DONE : create world (grid, objects, etc), place agent, place goal
        # Create an empty grid
        self.grid = Grid(
            width, height
        )  # optional, recreate the grid (it is alreayd defined in __init__ of super class)

        # Generate the surrounding walls
        self.grid.wall_rect(
            0, 0, width, height
        )  # important to avoid falling in void I think

        # Place a goal square in the bottom-right corner
        self.goal_coords = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_coords)

        # Place the agent
        self.place_agent()

    def step(self, action):

        # MUST BE DONE :
        # 1. update t and check if t >= max_steps
        # 2. deals with action
        # 3. render if render_mode is "human"
        # 4. generate observation
        # 5. return observation, reward, done, truncated, info

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        x, y = action
        x_true, y_true = self.goal_coords
        if x == x_true and y == y_true:
            reward = 1
        else:
            reward = 0
            self.failure_reason = "The goal is not at the given position."
        truncated = True
        done = True
        info = {}

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, done, truncated, info

    def get_new_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete([self.width, self.height])

    def get_feedback(self):
        if self.failure_reason is not None:
            return {"failure_reason": ErrorTrace(self.failure_reason)}
        else:
            return {}


def main():
    env = GiveGoalPositionEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
