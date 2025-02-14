from typing import Any, Dict, List
from agent.base_controller import Controller
from env.base_meta_env import Observation, ActionType


class MoveForwardController(Controller):
    """A controller that will always moves forward.
    
    Signatures :
        - __init__(self)
        - act(self) -> ActionType (always "forward")
        - has_finished(self) -> bool (always False)
    """

    def __init__(self):
        pass

    def act(self) -> ActionType:
        return "forward"

    def has_finished(self) -> bool:
        return False
