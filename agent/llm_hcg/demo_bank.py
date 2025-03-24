import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from core.task import TaskRepresentation
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union


class TransitionData:
    """Represents the data of a transition (task, controller, feedback) in the demo bank."""

    def __init__(
        self,
        task_repr: TaskRepresentation,
        code: str,
        feedback: Dict[str, Any],
    ):
        """Initialize the TransitionData object.

        Args:
            task_repr (TaskRepresentation): the task representation
            code (str): the code used to solve the task. It should contain the instanciation of a 'controller' variable of type Controller.
            feedback (Dict[str, Any]): the feedback of the transition, as a dictionnary mapping feedback fields to their values (success, metric, errors, ...)
        """
        self.task_repr = task_repr
        self.code = code
        self.feedback = feedback

    def __repr__(self):
        return (
            f"Task description: \n{self.task_repr}\n\n"
            f"Code: \n{self.code}\n\n"
            f"Feedback: \n{self.feedback}"
        )


class DemoBank:
    """
    This class contains a bank of demonstrations, i.e. transitions (task, controller, feedback),
    that happened during the agent's training.
    """

    def __init__(self, config_agent: Dict):
        """Initialize the demo bank.

        Args:
            config_agent (Dict): the agent configuration
        """
        print("Initializing demo bank...")
        # Extract config
        self.config_agent = config_agent
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
        return "Demo bank :" + "\n\n".join(repr(t) for t in self.transitions)
