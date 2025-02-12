import os
import re
from typing import Dict, List

from gym import Space
import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from core.task import TaskRepresentation
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union


class RandomController(Controller):

    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, observation: Observation) -> ActionType:
        action = self.action_space.sample()
        return action

    def has_finished(self):
        return False


class RandomAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

    def get_controller(self, task: TaskRepresentation) -> Controller:
        return RandomController(action_space=task.action_space)

    def update(
        self,
        task: TaskRepresentation,
        controller: Controller,
        feedback: Dict[str, Union[float, str]],
    ):
        pass
