import os
import re
from typing import Dict, List, Union
from abc import ABC, abstractmethod
import enum
import random

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from core.task import Task, TaskDescription


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

    def get_controller(self, task_description: TaskDescription) -> Controller:
        print(f"You are going to solve the following task: {task_description}")
        return HumanController(config=self.config)

    def update(
        self,
        task: Task,
        task_description: TaskDescription,
        controller: Controller,
        feedback: FeedbackAggregated,
    ):
        print(f"Feedback received: {feedback}")
