from abc import ABC, abstractmethod
from typing import Any, Dict, List
from env.base_meta_env import Observation, Action


class Controller(ABC):
    """A controller is an object that can act in an environment.

    Each controller should be expected to solve a specific task.

    The internal attributes of a controller can serve two purposes:
    - the caracteristics of the task that the controller must solve. Example : the coordinates of the goal for a navigation task.
    - the internal memory of the controller, to allow for time-dependant actions. Example : for a task consisting of doing a loop around a wall, the controller necessarily needs to remember where he is in the process.

    Also, the controller should be able to tell when it has finished its task at any time, which will necessarily requires access to the task and the internal state of the controller (its memory/view of the world).

    A sub-class of controller is expected to implement the act() method, in which it should return an action but also update its internal state,
    and the has_finished() method, in which it should return whether the controller has finished its task.
    """

    def __init__(self, **task_information: Dict[str, Any]):
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

    @abstractmethod
    def has_finished(self) -> bool:
        """Return whether the controller has finished its task.

        Returns:
            bool: whether the controller has finished its task.
        """
        raise NotImplementedError
