import os
import re
from typing import Dict, List

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller, Task
from env.base_meta_env import BaseMetaEnv, Observation, Action, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union


class HumanController(Controller):
    
    def act(self, observation: Observation) -> Action:
        print(f"Current observation: {observation}")
        action = input("What action do you want to take ?")
        return action
    
    def has_finished(self):
        print("Does the controller has finished its task ?")
        return input("True/False") == "True"
    
class HumanAgent(BaseAgent):
    
    def get_controller(self, task: Task) -> Controller:
        print(f"You are going to solve the following task: {task}")
        return HumanController()
    
    def update(self, task, controller, feedback):
        print(f"Feedback received: {feedback}")