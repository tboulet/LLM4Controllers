from typing import Callable, Tuple, Type, Union, Dict, Any, List, Optional
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX


dict_actions: Dict[str, Tuple[int, str]] = {
    "left": (
        0,
        "Turn the direction of the agent to the left (don't move in that direction)",
    ),
    "right": (
        1,
        "Turn the direction of the agent to the right (don't move in that direction)",
    ),
    "forward": (2, "Move one tile forward in the direction the agent is facing"),
    "pickup": (
        3,
        "Pick up the object the agent is facing (if any) and add it to the agent's inventory",
    ),
    "drop": (
        4,
        "Drop the object from the agent's inventory (if any) in front of the agent",
    ),
    "toggle": (5, "Toggle/activate an object in front of the agent"),
    "done": (6, "End the episode if the task is completed"),
}

dict_directions: Dict[str, int] = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}