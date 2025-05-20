from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List, Optional
from gymnasium import Space

from core.types import ActionType, Observation, InfoDict


class TaskDescription:
    """The information about a specific task that the controller must solve.
    It should contain any information related to the caracteristics of the task.
    For example, for a navigation task, the task information should contain at least the coordinates of the goal.
    """

    def __init__(
        self,
        description: str = None,
        observation_space: Space = None,
        action_space: Space = None,
    ) -> None:
        """Initialize the TaskDescription object.

        Args:
            description (str): a textual description of the task, for example "go to the green ball" (can be the same as the name)
            observation_space (Space): the observation space of the environment
            action_space (Space): the action space of the environment
        """
        self.description = description
        self.observation_space = observation_space
        self.action_space = action_space

    def __repr__(self) -> str:
        list_strings = []
        if self.description is not None:
            list_strings.append(f"{self.description}")
        if self.observation_space is not None:
            list_strings.append(f"Observation gym space : \n{self.observation_space}")
        if self.action_space is not None:
            list_strings.append(f"Action gym space (the actions you can take MUST belong to this space) : {self.action_space}\nYou HAVE to take an action that belongs to this space.")
        return "\n\n".join(list_strings)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskDescription):
            return False
        return (
            self.description == other.description
            and self.observation_space == other.observation_space
            and self.action_space == other.action_space
        )


class Task(ABC):
    """A task is a specific instance of a problem that the agent must solve."""

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the task.

        Returns:
            str: the name of the task
        """
    
    @abstractmethod
    def get_description(self) -> TaskDescription:
        """Get the task representation.

        Returns:
            TaskDescription: the task representation
        """

    @abstractmethod
    def get_code_repr(self) -> str:
        """Get the code representation of the task."""
        
        
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[Observation, InfoDict]:
        """Reset the task to its initial state.

        Returns:
            Observation: the initial observation of the task
            InfoDict: additional information about the current state of the task
        """

    @abstractmethod
    def step(
        self,
        action: ActionType,
    ) -> Tuple[
        Observation,
        float,
        bool,
        bool,
        InfoDict,
    ]:
        """Take a step in the task.

        Args:
            action (ActionType): the action to take

        Returns:
            Tuple[Observation, float, bool, bool, InfoDict]: the observation, reward, done, truncated and info
        """
        pass
    
    def render(self) -> None:
        """Render the task.
        """
        pass

    def close(self) -> None:
        """Close the task."""
        pass
    
    def get_feedback(self) -> Dict[str, Any]:
        """Get feedback from the task.

        Returns:
            Dict[str, Any]: the feedback from the task
        """
        return {}