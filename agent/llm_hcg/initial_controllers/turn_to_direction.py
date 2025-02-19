from typing import Any, Dict, List
from agent.base_controller import Controller
from env.base_meta_env import Observation, ActionType


class TurnToDirectionController(Controller):
    """A controller that will turn to a specific direction (0: up, 1: right, 2: down, 3: left).
    
    Signatures :
        - __init__(self, direction_target: int)
        - act(self, observation) -> ActionType (in ["left", "right", "done"])
        - has_finished(self) -> bool
    """

    def __init__(self, direction_target: int):
        assert direction_target in [0, 1, 2, 3]
        self.direction_target = direction_target
        self.do_has_finished = False
        
    def act(self, observation: Observation) -> ActionType:
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
        return self.do_has_finished
