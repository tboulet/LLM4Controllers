from abc import ABC, abstractmethod
from typing import Any, Dict, List
from env.base_meta_env import Observation, ActionType


class Controller(ABC):
    """A controller is an object that can act in an environment.
    Each controller should be expected to solve a specific task.

    A sub-class of controller is expected to implement :
    - the __init__(**kwargs) method to initialize the controller with eventually task-related informations (e.g. the coordinates of the goal for a navigation task).
    - the act(Observation) -> Action method, in which it should return an action but can also update its internal state.
    - the has_finished() -> bool method to indicate whether the controller has finished its task. If the controller has no notion of task completion or if it unclear, you can return False by default.

    The internal attributes of a controller can serve two purposes:
    - the caracteristics of the task that the controller must solve. Example : the coordinates of the goal for a navigation task.
    - the internal memory of the controller, to allow for time-dependant actions. Example : for a task consisting of doing a loop around a wall, the controller necessarily needs to remember where he is in the process.
    """

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        """Initialize the controller with 
        
        Args:
            kwargs (Dict[str, Any]): a dictionary of parameters to initialize the controller.
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: Observation) -> ActionType:
        """Act in the environment using the given observation.

        Args:
            observation (Observation): the observation of the environment.

        Returns:
            Action: the action to take in the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def has_finished(self) -> bool:
        """Return whether the controller has finished its task.

        Returns:
            bool: whether the controller has finished its task.
        """
        raise NotImplementedError
