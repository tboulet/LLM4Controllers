
from typing import Optional
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class AutoSuccessMGEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: Optional[int] = None,
        **kwargs,
    ):

        mission_space = MissionSpace(mission_func=self._gen_mission)

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
    def _gen_mission():
        return "do nothing particular"
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0
            
            
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        terminated = True
        reward = self._reward()
        return obs, reward, terminated, truncated, info


def main():
    env = AutoSuccessMGEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()