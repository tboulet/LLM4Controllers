import os
import re
from typing import Dict, List

from gymnasium import Space
import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from agent.base_agent2 import BaseAgent2
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from core.play import play_controller_in_task
from core.task import Task, TaskDescription
from core.utils import sanitize_name
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
        self.n_episodes_eval = config["n_episodes_eval"]
        self.tasks = self.env.get_current_tasks()
        self.tasks = sorted(self.tasks, key=lambda task: str(task))
        self.iter: int = 0

    def step(self):
        for task in self.tasks:
            task_description = task.get_description()
            print(f"Task received: {task}.\n")
            controller = RandomController(action_space=task_description.action_space)
            feedback_over_eps = play_controller_in_task(
                controller,
                task,
                n_episodes=self.n_episodes_eval,
                is_eval=False,
                log_subdir=f"task_{sanitize_name(str(task))}",
            )
            feedback_over_eps.aggregate()
            # Log the metrics
            self.log_as_texts(
                {
                    f"feedback_over_eps.txt": feedback_over_eps.get_repr(),
                },
                log_subdir=f"task_{sanitize_name(str(task))}",
            )
            feedback_agg_over_controllers = (
                FeedbackAggregated()
            )  # aggregate on one controller just to have metric name consistency with CG
            metrics_agg_over_episodes = feedback_over_eps.get_metrics()
            feedback_agg_over_controllers.add_feedback(metrics_agg_over_episodes)
            feedback_agg_over_controllers.aggregate()
            self.logger.log_scalars(
                feedback_agg_over_controllers.get_metrics(prefix=sanitize_name(str(task))), step=0
            )

            # Log runtime metrics
            metrics_runtime = self.get_runtime_metrics()
            self.logger.log_scalars(metrics_runtime, step=0)

            # Move forward the iter counter
            self.iter += 1

    def is_done(self):
        return self.iter >= 1
