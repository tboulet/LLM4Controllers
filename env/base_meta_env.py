from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List, Optional
from gymnasium import Space
import numpy as np

from core.task import Task, TaskRepresentation
from core.types import Observation, ActionType, InfoDict
from core.spaces import FiniteSpace


class BaseMetaEnv(ABC):
    def __init__(self, config: Dict) -> None:
        self.config = config

    @abstractmethod
    def get_textual_description(self) -> str:
        """Return a textual description of the environment, this will be given to the agent at the beginning of the training
        to help it getting a general idea of the environment and the tasks it will be facing.

        It should include the general principle of the environment, the actions available, the structure of the observations, the reward system, etc.

        Returns:
            str: the textual description of the environment
        """

    @abstractmethod
    def reset(self, seed: Union[int, None] = None, **kwargs) -> Tuple[Observation, Task, TaskRepresentation, Dict[str, Any]]:
        """Reset the environment to its initial state and starts a new episode.
        Returns the first observation of the episode as well as the textual description of the task in particular.
        Also returns a dictionary containing additional information about the environment.

        This method can return, if possible, the optimal reward achievable in the episode for the task (e.g. the reward corresponding to the shortest path to the goal).

        Args:
            seed (Union[int, None], optional): the seed to use for the random number generator. Defaults to None (random seed).

        Returns:
            Observation: the first observation of the episode
            Task: the "task" object that the agent will have to solve. This should be interpreted as an identifier of the task more strict than the textual description.
            TaskRepresentation: the description of the task in particular. This should involve a textual description of the task but also its caracteristics (e.g. coordinate of the goal, etc.)
            InfoDict: a dictionary containing additional information about the environment
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionType) -> Tuple[Observation, float, bool, InfoDict]:
        """Take a step in the environment using the given action and returns the new observation, the reward, whether the episode is over and additional information.

        Args:
            action (Any): the action to take in the environment

        Returns:
            Observation: the new observation
            float: the reward obtained after taking the action
            bool: whether the episode is over
            InfoDict: a dictionary containing additional information about the environment
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, task : TaskRepresentation, feedback : Dict[str, Any]) -> None:
        """Update the environment based on the feedback received from the agent.

        Args:
            task (TaskRepresentation): the description of the task the agent was performing
            feedback (Dict[str, Any]): a dictionary containing the feedback from the agent. It should contain at least the following:
                - "success" (bool): whether the agent has successfully completed the task
                - "reward" (float): the reward received by the agent
                - (optional) "error" (str): the error message if an error occured during the agent execution
        """
        raise NotImplementedError
    
    def get_feedback(self) -> Dict[str, Any]:
        """Return additional feedback about the environment that can be useful for the agent.

        Returns:
            Dict[str, Any]: a dictionary containing additional feedback about the environment
        """
        return {}
    
    def render(self):
        """Render the environment"""
        pass

    def close(self):
        """Close the environment"""
        pass
