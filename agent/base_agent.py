from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from agent.base_controller import Controller
from core.feedback_aggregator import FeedbackAggregated
from env.base_meta_env import Observation, ActionType
from core.task import Task, TaskDescription


class BaseAgent(ABC):

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def get_controller(self, task_description: TaskDescription) -> Controller:
        """Get the controller for the given task description.

        Args:
            task_description (Task): the task description.

        Returns:
            Controller: the controller for the task.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        task: Task,
        task_description: TaskDescription,
        controller: Controller,
        feedback: FeedbackAggregated,
    ):
        """Update the agent's internal state (library, knowledges, etc.) based on the feedback received from the environment.

        Args:
            task (Task): the task description of the task the controller was performing.
            task_description (TaskDescription): the task description of the task the controller was performing.
            controller (Controller): the controller that was performing the task. It may be useful to keep the code version of the controller as an internal variable of the agent.
            feedback (FeedbackAggregated): the feedback received from the environment. It may be useful to keep the feedback as an internal variable of the agent.
        """
        raise NotImplementedError

    def give_textual_description(self, description: str):
        """Give a textual description of the environment to the agent.

        Args:
            description (str): the textual description of the environment.
        """
        pass  # Pass for now to also allow for agents that do not need this information.
