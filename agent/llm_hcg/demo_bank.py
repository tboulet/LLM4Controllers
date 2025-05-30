import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from core.feedback_aggregator import FeedbackAggregated
from core.task import Task, TaskDescription
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union

from agent.llm_hcg.graph_viz import ControllerVisualizer


class TransitionData:
    """Represents the data of a transition (task, controller, feedback) in the demo bank."""

    def __init__(
        self,
        task: Task,
        task_repr: TaskDescription,
        code: str,
        feedback: FeedbackAggregated,
    ):
        """Initialize the TransitionData object.

        Args:
            task (Task): the task object
            task_repr (TaskRepresentation): the task representation
            code (str): the code used to solve the task. It should contain the instanciation of a 'controller' variable of type Controller.
            feedback (FeedbackAggregated): the feedback of the transition
        """
        self.task = task
        self.task_repr = task_repr
        self.code = code
        self.feedback = feedback

    def __repr__(self):
        return (
            f"Task: \n{self.task}\n\n"
            f"Task description: \n{self.task_repr}\n\n"
            f"Code: \n```python\n{self.code}\n```\n\n"
            f"Feedback: \n{self.feedback}"
        )


class DemoBank:
    """
    This class contains a bank of demonstrations, i.e. transitions (task, controller, feedback),
    that happened during the agent's training.
    """

    def __init__(self, config_agent: Dict, visualizer: ControllerVisualizer):
        """Initialize the demo bank.

        Args:
            config_agent (Dict): the agent configuration
        """
        print("Initializing demo bank...")
        # Extract config
        self.config_agent = config_agent
        self.visualizer = visualizer
        # Initialize the demo bank
        self.transitions: List[TransitionData] = []

    def add_transition(self, transition: TransitionData):
        """Add a transition to the demo bank.

        Args:
            transition (TransitionData): the transition data
        """
        self.transitions.append(transition)

    def sample_transitions(
        self, n_transitions: int, method: str, **kwargs
    ) -> List[TransitionData]:
        """Sample transitions from the demo bank according to a given method.

        Args:
            n_transitions (int): the number of transitions to sample
            method (str): the sampling method to use
            **kwargs: additional arguments for the sampling method

        Returns:
            List[TransitionData]: a list of sampled transitions
        """
        if method == "uniform":
            # Sample uniformly up to n_transitions transitions, less if there are less transitions in the bank
            transitions = random.sample(
                self.transitions, k=min(n_transitions, len(self.transitions))
            )
            return transitions
        else:
            raise NotImplementedError(f"Sampling method {method} not implemented.")

    def __repr__(self):
        if len(self.transitions) == 0:
            return "The Demo bank is empty for now.\n"
        return f"Demo bank :\n\n" + "\n\n".join(str(t) for t in self.transitions)
