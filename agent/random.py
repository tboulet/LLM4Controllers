import os
import re
from typing import Dict, List

from gymnasium import Space
import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from agent.base_agent2 import BaseAgent2
from core.loggers.base_logger import BaseLogger
from core.play import play_controller_in_task
from core.task import Task, TaskDescription
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

    def __init__(self, config: Dict, logger: BaseLogger = None):
        self.config = config
        self.logger = logger

    def get_controller(self, task_description: TaskDescription) -> Controller:
        return RandomController(action_space=task_description.action_space)

    def update(
        self,
        task: Task,
        task_description: TaskDescription,
        controller: Controller,
        feedback: Dict[str, Union[float, str]],
    ):
        pass


class RandomAgent2(BaseAgent2):

    def __init__(self, config, logger: BaseLogger, env: BaseMetaEnv):
        super().__init__(config, logger, env)
        self.tasks = self.env.get_current_tasks()
        self.tasks = sorted(self.tasks, key=lambda task: str(task))
        self.t = 0

    def step(self):
        task = self.tasks[self.t]
        print(f"Step {self.t}, task received: {task}")
        task_description = task.get_description()
        controller = RandomController(action_space=task_description.action_space)
        feedback_agg = play_controller_in_task(
            controller, task, n_episodes=10, is_eval=False, log_dir=f"task_{self.t}"
        )
        feedback_agg.aggregate()
        # Log the metrics
        self.log_texts(
            {
                f"feedback.txt": feedback_agg.get_repr(),
            },
            log_dir=f"task_{self.t}",
        )
        self.logger.log_scalars(
            feedback_agg.get_metrics(prefix=task),
            step=self.t,
        )
        self.t += 1

    def is_done(self):
        return self.t >= len(self.tasks)
