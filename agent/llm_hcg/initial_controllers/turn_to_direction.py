from typing import Any, Dict, List
from agent.base_controller import Controller
from env.base_meta_env import Observation, ActionType


class TurnToDirectionController(Controller):
    """A controller that will turn to a specific direction (0: up, 1: right, 2: down, 3: left).
    It uses the 'left' and 'right' actions to turn towards the desired direction.
    """

    def __init__(self, direction_target: int):
        """Initialize the controller with the target direction.

        Args:
            direction_target (int): the target direction to turn to (0: up, 1: right, 2: down, 3: left).
        """
        assert direction_target in [0, 1, 2, 3]
        self.direction_target = direction_target
        self.do_has_finished = False

    def act(self, observation: Observation) -> ActionType:
        """Return a 'left' or 'right' action to turn towards the target direction.
        It is done by comparing the current direction with the target direction.
        If the controller direction is already the target direction, it returns 'done' (no action).

        Args:
            observation (Observation): the observation from the environment.

        Returns:
            ActionType: the action to take in the environment.
        """
        direction = observation["direction"]
        if direction == self.direction_target:
            self.do_has_finished = True
            return "done"
        elif direction == (self.direction_target + 1) % 4:
            self.do_has_finished = True
            return "left"
        elif direction == (self.direction_target - 1) % 4:
            self.do_has_finished = True
            return "right"
        else:
            self.do_has_finished = False
            return "right"

    def has_finished(self) -> bool:
        """Returns whether the controller has finished.
        It is the case whenever the last observed observation has the direction equal to the target direction.

        Returns:
            bool: whether the controller has finished.
        """
        return self.do_has_finished
