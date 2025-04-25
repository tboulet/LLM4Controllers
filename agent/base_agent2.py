from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from agent.base_controller import Controller
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from env.base_meta_env import BaseMetaEnv, Observation, ActionType
from core.task import Task, TaskDescription


class BaseAgent2(ABC):

    def __init__(self, config: Dict, logger : BaseLogger, env: BaseMetaEnv):
        self.config = config
        self.logger = logger
        self.env = env
    
    @abstractmethod
    def step(self):
        """Perform a step of the agent."""
        raise NotImplementedError