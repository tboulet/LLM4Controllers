from agent.base_controller import Controller, Observation, ActionType
from ..utils_PCs import MoveForwardController, TurnToDirectionController


class SpecializedController(Controller):
    """A controller that will turn to north (0: up) and then move forward."""

    def __init__(self):
        self.turn_to_north_controller = TurnToDirectionController(direction_target=0)
        self.move_forward_controller = MoveForwardController()

    def act(self, observation: Observation) -> ActionType:
        """Turn to north until the agent is facing north, then move forward."""
        if not self.turn_to_north_controller.has_finished():
            return self.turn_to_north_controller.act(observation)
        return self.move_forward_controller.act(observation)

    def has_finished(self) -> bool:
        """Return False as the controller never finishes.

        Returns:
            bool: Whether the controller has finished.
        """
        return False

controller = SpecializedController()