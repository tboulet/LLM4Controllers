from abc import ABC, abstractmethod
from typing import Any, Dict, List
from agent.base_controller import Controller
from env.base_meta_env import Observation, Action


    
    
    
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
