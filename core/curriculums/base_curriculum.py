from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Tuple, Type, Union, Set
from core.task import TaskRepresentation
import numpy


class BaseCurriculum(ABC):
    """A curriculum model a dynamical task distribution that is updated depending on the performance of the agent.
    Initially, only very basic tasks are part of its distribution, and more complex tasks are added as the agent's performance improves.
    """

    @abstractmethod
    def sample(self) -> TaskRepresentation:
        """Sample an task from the curriculum distribution."""

    @abstractmethod
    def update(self, task: TaskRepresentation, feedback: Dict[str, Any]):
        """Update the curriculum based on the feedback received from the agent."""
