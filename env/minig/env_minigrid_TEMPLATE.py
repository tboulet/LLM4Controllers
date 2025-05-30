from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from core.types import ErrorTrace


class AutoSuccessMGEnv(MiniGridEnv):
    # SHOULD BE DONE : docstring

    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        # Optional things implemented here
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        if max_steps is None:
            max_steps = 4 * size**2

        # MUST BE DONE : create mission space
        mission_space = MissionSpace(mission_func=self._gen_mission)

        # CAN BE DONE : define obs (with or without mission) and action spaces

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
        return "Get to the green goal square"

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

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())

        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def step(self, action):

        if False:
            return super().step(action)  # normal step

        # MUST BE DONE (note : super().step() does all 4):
        # 1. update t and check if t >= max_steps
        # 2. deals with action
        # 3. render if render_mode is "human"
        # 4. generate observation
        # 5. return observation, reward, done, truncated, info

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        deals_with_action()

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, done, truncated, info

    # CAN BE DONE : define an alternative action space to the default one
    def get_new_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete([self.width, self.height])

    # CAN BE DONE : define a get_feedback function that add additional env-based feedback to F_i
    def get_feedback(self):
        if self.failure_reason is not None:
            return {"failure_reason": ErrorTrace(self.failure_reason)}
        else:
            return {}


def main():
    env = AutoSuccessMGEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
