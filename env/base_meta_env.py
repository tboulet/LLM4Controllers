from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List, Optional
from gym import Space
import numpy as np

from core.types import Observation, Action, InfoDict
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
    def reset(seed: Union[int, None] = None) -> Tuple[Observation, str, Dict[str, Any]]:
        """Reset the environment to its initial state and starts a new episode.
        Returns the first observation of the episode as well as the textual description of the task in particular.
        Also returns a dictionary containing additional information about the environment.

        This method can return, if possible, the optimal reward achievable in the episode for the task (e.g. the reward corresponding to the shortest path to the goal).

        Args:
            seed (Union[int, None], optional): the seed to use for the random number generator. Defaults to None (random seed).

        Returns:
            Observation: the first observation of the episode
            str: the textual description of the task in particular. This should involve a textual description of the task but also its caracteristics (e.g. coordinate of the goal, etc.)
            InfoDict: a dictionary containing additional information about the environment
        """
        raise NotImplementedError

    @abstractmethod
    def step(action: Any) -> Tuple[Observation, float, bool, InfoDict]:
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

    def render(self):
        """Render the environment"""
        pass

    def close(self):
        """Close the environment"""
        pass
