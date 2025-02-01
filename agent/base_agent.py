from abc import ABC, abstractmethod
from typing import Any, Dict, List
from env.base_meta_env import BaseMetaEnv, Observation, Action, InfoDict
import numpy as np


class Memory(Dict[str, Any]):
    """The memory of a controller.
    It should contain any information that the controller needs to store between steps.
    For example, the memory could be used to store the previous action taken by the controller.
    """


class TaskInformation(Dict[str, Any]):
    """The information about a specific task that the controller must solve.
    It should contain any information related to the caracteristics of the task.
    For example, for a navigation task, the task information should contain at least the coordinates of the goal.
    """
    
    
class Controller(ABC):
    """A controller is simply an object that can act in an environment."""
    
    def __init__(self, **info_task: TaskInformation):
        pass
    
    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Act in the environment using the given observation.
        
        Args:
            observation (Observation): the observation of the environment.
        
        Returns:
            Action: the action to take in the environment.
        """
        raise NotImplementedError
    
class Task:
    """A task is a description of a specific problem that the controller must solve.
    It can be textual, but it could also be under other form, such as goal embeddings, etc.
    """
    pass

    
class BaseAgent(ABC):

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def get_controller(self, task: Task) -> Controller:
        """Get the controller for the given task description.
        
        Args:
            task (Task): the task description.
        
        Returns:
            Controller: the controller for the task.
        """
        raise NotImplementedError
    
    def give_textual_description(self, description: str):
        """Give a textual description of the environment to the agent.
        
        Args:
            description (str): the textual description of the environment.
        """
        pass # Pass for now to also allow for agents that do not need this information.
