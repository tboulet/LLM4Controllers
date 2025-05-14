from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Union
from agent.base_controller import Controller
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from env.base_meta_env import BaseMetaEnv, Observation, ActionType
from core.task import Task, TaskDescription
from tbutils.tmeasure import RuntimeMeter

class BaseAgent2(ABC):

    def __init__(self, config: Dict, logger: BaseLogger, env: BaseMetaEnv):
        self.config = config
        self.logger = logger
        self.env = env
        config_logs = self.config["config_logs"]
        log_dir = config_logs["log_dir"]
        self.list_log_dirs_global: List[str] = []
        if config_logs["do_log_on_new"]:
            self.list_log_dirs_global.append(os.path.join(log_dir, config["run_name"]))
        if config_logs["do_log_on_last"]:
            self.list_log_dirs_global.append(os.path.join(log_dir, "last"))

    @abstractmethod
    def step(self):
        """Perform a step of the agent."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Check if the agent is done."""
        return False

    def log_texts(
        self,
        dict_name_to_text: Dict[str, str],
        log_dir: str,
    ):
        """Log texts in a directory. For each (key, value) in the directory, the file <log_dir>/task_<t>/<key> will contain the value.
        It will do that for all <log_dir> in self.list_run_names.

        Args:
            dict_name_to_text (Dict[str, str]): a mapping from the name of the file to create to the text to write in it.
        """
        with RuntimeMeter("log_texts"):
            list_log_dirs: List[str] = []
            for log_dir_global in self.list_log_dirs_global:
                log_dir = os.path.join(log_dir_global, log_dir)
                list_log_dirs.append(log_dir)

            for log_dir in list_log_dirs:
                os.makedirs(log_dir, exist_ok=True)
                for name, text in dict_name_to_text.items():
                    assert isinstance(name, str) and isinstance(
                        text, str
                    ), "Keys and values of dict_name_to_text should be strings."
                    log_file = os.path.join(log_dir, name)
                    with open(log_file, "w") as f:
                        f.write(text)
                    f.close()
                    print(f"[LOGGING] : Logged {log_file}")
