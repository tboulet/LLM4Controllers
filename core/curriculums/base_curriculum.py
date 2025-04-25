from abc import ABC, abstractmethod
import random
from typing import Any, Dict, Generic, List, TypeVar

from core.feedback_aggregator import FeedbackAggregated

Objective = TypeVar("Objective")


class BaseCurriculum(ABC, Generic[Objective]):
    """A curriculum models a dynamical objective distribution that is updated
    depending on the performance of the agent.

    Initially, only very basic objectives are part of its distribution, and
    more complex objectives are added as the agent's performance improves.
    """

    @abstractmethod
    def sample(self) -> Objective:
        """Sample an objective from the curriculum distribution."""

    @abstractmethod
    def get_current_objectives(self) -> List[Objective]:
        """Get all the objectives that are currently available to the agent.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, objective: Objective, feedback: FeedbackAggregated):
        """Update the curriculum based on the feedback received from the agent."""
