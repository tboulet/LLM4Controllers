from typing import Dict, List

import numpy as np
from agent.base_agent import BaseAgent, Controller, Memory
from env.base_meta_env import BaseMetaEnv, Observation, Action, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union



class RandomController(Controller):
    """A controller that acts randomly in the environment."""

    def act(self, observation: Observation) -> Action:
        return random.choice(
                [
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                ]
            )


class LLMBasedHierarchicalControllerGenerator(BaseAgent):

    def __init__(self, config: Dict):
        super().__init__(config)
        
    def give_textual_description(self, description: str):
        print(description)
        
    def get_controller(self, task_description: str) -> Controller:
        return RandomController()
    
