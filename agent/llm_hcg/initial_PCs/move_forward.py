class MoveForwardController(Controller):
    """A controller that will always moves forward."""

    def __init__(self):
        pass

    def act(self, observation: Observation) -> ActionType:
        """Return a 'forward' action to move forward.

        Returns:
            ActionType: The action to take in the environment.
        """
        return "forward"
