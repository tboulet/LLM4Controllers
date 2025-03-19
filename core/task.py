from typing import Tuple, Union, Dict, Any, List, Optional
from gymnasium import Space


class Task:
    """A task is a specific instance of a problem that the agent must solve.
    It can be seen as a specific instance of a task representation.
    However it can be not absolutely specific, for example the goal can be a random point.
    """
    pass

class TaskRepresentation:
    """The information about a specific task that the controller must solve.
    It should contain any information related to the caracteristics of the task.
    For example, for a navigation task, the task information should contain at least the coordinates of the goal.
    """

    def __init__(
        self,
        name: str,
        description: str = None,
        observation_space: Space = None,
        action_space: Space = None,
    ) -> None:
        """Initialize the TaskRepresentation object.

        Args:
            name (str): the name (unique identifier) of the task, for exemple "go to the green ball"
            description (str): a textual description of the task, for example "go to the green ball" (can be the same as the name)
            observation_space (Space): the observation space of the environment
            action_space (Space): the action space of the environment
        """
        self.name = name
        self.description = description
        self.observation_space = observation_space
        self.action_space = action_space

    def __repr__(self) -> str:
        res = ""
        if self.name is not None:
            res = f"Task name : {self.name}."
        if self.description != self.name and self.description is not None:
            res += f"\nTask description : {self.description}."
        if self.observation_space is not None:
            res += f"\nObservation gym space : {self.observation_space}."
        if self.action_space is not None:
            res += f"\nAction gym space (the actions you can take MUST belong to this space) : {self.action_space}. You HAVE to take an action that belongs to this space."
        return res