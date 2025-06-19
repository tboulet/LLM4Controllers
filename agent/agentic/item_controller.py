

from abc import ABC, abstractmethod

from core.types import Observation, ActionType


class Controller(ABC):
    """
    Base class for all controllers in the agentic framework.
    
    A controller is a component that interacts with an RL-like task by implementing the `act` method.
    
    It is possible to store information in the controller by using the instance's attributes.
    This can be usefull for task requiring memory over the episode, or for gathering information about the task for later analysis.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the controller with any additional parameters.
        
        Args:
            **kwargs: Additional parameters to initialize the controller.
        """
        
    @abstractmethod
    def act(self, observation : Observation) -> ActionType:
        """
        Perform an action based on the given observation.
        
        Args:
            observation (Observation): The current observation from the environment.
            
        Returns:
            ActionType: The action to be performed in response to the observation.
        """