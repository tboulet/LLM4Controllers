from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
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
    
    @abstractmethod
    def update(self, task : Task, controller : Controller, feedback : Dict[str, Union[float, str]]):
        """Update the agent's internal state (library, knowledges, etc.) based on the feedback received from the environment.

        Args:
            task (Task): the task description of the task the controller was performing.
            controller (Controller): the controller that was performing the task. It may be useful to keep the code version of the controller as an internal variable of the agent.
            feedback (Dict[str, Union[float, str]]): a dictionnary containing the feedback from the environment. It should contain at least the following:
                - "success" (bool): whether the controller has successfully completed the task.
                - "reward" (float): the reward received by the controller.
                - (optinal) "error" (str): the error message if an error occured during the controller execution 
        """
        raise NotImplementedError
    
    def give_textual_description(self, description: str):
        """Give a textual description of the environment to the agent.
        
        Args:
            description (str): the textual description of the environment.
        """
        pass # Pass for now to also allow for agents that do not need this information.
