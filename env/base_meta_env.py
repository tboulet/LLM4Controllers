from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List, Optional
from gymnasium import Space
import numpy as np

from core.loggers.base_logger import BaseLogger
from core.task import Task, TaskDescription
from core.types import Observation, ActionType, InfoDict
from core.spaces import FiniteSpace


class BaseMetaEnv(ABC):
    def __init__(self, config: Dict, logger: BaseLogger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def get_task(self) -> Task:
        """Get a new task for the agent to solve.

        Returns:
            Task: the task to solve
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, task: Task, feedback: Dict[str, Any]) -> None:
        """Update the environment based on the feedback received from the agent.

        Args:
            task (Task): the task that was solved by the agent
            feedback (Dict[str, Any]): a dictionary containing the feedback from the agent. It should contain at least the following:
                - "success" (bool): whether the agent has successfully completed the task
                - "reward" (float): the reward received by the agent
                - (optional) "error" (str): the error message if an error occured during the agent execution
        """
        raise NotImplementedError

    @abstractmethod
    def get_textual_description(self) -> str:
        """Return a textual description of the environment, this will be given to the agent at the beginning of the training
        to help it getting a general idea of the environment and the tasks it will be facing.

        It should include the general principle of the environment, the actions available, the structure of the observations, the reward system, etc.
        
        Returns:
            str: the textual description of the environment
        """
        raise NotImplementedError
