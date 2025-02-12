import os
import re
from typing import Dict, List
from abc import ABC, abstractmethod
import enum
import random

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from core.task import TaskRepresentation

class HumanController(Controller):

    def __init__(self, config: Dict):
        self.config = config

    def act(self, observation: Observation) -> ActionType:
        if self.config["do_print_observation"]:
            print(f"Current observation: {observation}")
        action = input("What action do you want to take ?")
        return action

    def has_finished(self):
        print("Does the controller has finished its task ?")
        return input("True/False") == "True"


class HumanAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
        
    def get_controller(self, task: TaskRepresentation) -> Controller:
        breakpoint()
        print(f"You are going to solve the following task: {task}")
        return HumanController(config = self.config)

    def update(self, task, controller, feedback):
        print(f"Feedback received: {feedback}")
