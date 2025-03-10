from typing import Tuple, Union, Dict, Any, List, Optional
from gymnasium import Space


class TaskRepresentation:
    """The information about a specific task that the controller must solve.
    It should contain any information related to the caracteristics of the task.
    For example, for a navigation task, the task information should contain at least the coordinates of the goal.
    """

    def __init__(
        self,
        name: str,
        family_task: str,
        description: str,
        observation_space: Space,
        action_space: Space,
        kwargs: Dict[str, Any] = None,
    ) -> None:
        """Initialize the TaskRepresentation object.

        Args:
            name (str): the name (unique identifier) of the task, for exemple "go to the green ball"
            family_task (str): the family to which this task belong, for example "go to the <color> <obj_type>"
            description (str): a textual description of the task, for example "go to the green ball" (can be the same as the name)
            observation_space (Space): the observation space of the environment
            action_space (Space): the action space of the environment
            kwargs (Dict[str, Any], optional): the values of the placeholders in the family_task. Defaults to None. For example {"color": "green", "obj_type": "ball"} for the task "go to the green ball".
        """
        self.name = name
        self.family_task = family_task
        self.description = description
        self.observation_space = observation_space
        self.action_space = action_space
        self.kwargs = kwargs

    def __repr__(self) -> str:
        res = f"{self.name}."
        if self.name != self.family_task:
            res += f"\nThis task belongs to the family of task : {self.family_task}."
        if self.description != self.name:
            res += f"\nTask description : {self.description}."
        res += f"\nThe observation you will receive belong to the following gym space : {self.observation_space}."
        res += f"\nThe actions you can take MUST belong to the following gym space : {self.action_space}. You HAVE to take an action that belongs to this space."
        return res