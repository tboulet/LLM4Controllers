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


class Transition:
    """Represents the data of a transition (task, controller, feedback) in the demo bank."""

    def __init__(
        self,
        task_repr: TaskRepresentation,
        code: str,
        feedback: Dict[str, Any],
    ):
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
        print("Initializing demo bank...")
        # Extract config
        self.config = config_agent["config_demobank"]
        self.n_inference = self.config.get("n_inference", 5)
        self.n_training = self.config.get("n_training", 100)
        self.method_inference_sampling = self.config.get("method_inference_sampling", "uniform")
        # Initialize the demo bank
        self.transitions: List[Transition] = []

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)

    def sample_transitions(self, task_description : TaskRepresentation) -> List[Transition]:
        if self.method_inference_sampling == "uniform":
            # Sample uniformly up to n_inference transitions, less if there are less transitions in the bank
            transitions = random.sample(self.transitions, k=min(self.n_inference, len(self.transitions)))
            return transitions
        else:
            raise NotImplementedError(f"Sampling method {self.method_inference_sampling} not implemented.")