from typing import Any, Dict, List
from agent.base_controller import Controller
from env.base_meta_env import Observation, ActionType


class MoveForwardController(Controller):
    """A controller that will always moves forward."""

    def __init__(self):
        pass

    def act(self) -> ActionType:
        """Return a 'forward' action to move forward.

        Returns:
            ActionType: The action to take in the environment.
        """
        return "forward"

    def has_finished(self) -> bool:
        """Return False as the controller never finishes.

        Returns:
            bool: Whether the controller has finished.
        """
        return False
