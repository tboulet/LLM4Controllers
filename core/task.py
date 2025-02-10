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
        description: str,
        observation_space: Space,
        action_space: Space,
        variables: Dict[str, Any] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.observation_space = observation_space
        self.action_space = action_space
        self.variables = variables
        
    def __repr__(self) -> str:
        return f"""TaskRepresentation(
            name = {self.name},
            description = {self.description},
            observation_space = {self.observation_space},
            action_space = {self.action_space},
            variables = {self.variables}
        )"""