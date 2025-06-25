from abc import ABC, abstractmethod
import json
import os
from typing import Any, Callable, Dict, List, Union
from agent.base_controller import Controller
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from env.base_meta_env import BaseMetaEnv, Observation, ActionType
from core.task import Task, TaskDescription
from tbutils.tmeasure import RuntimeMeter
from tbutils.tmeasure import RuntimeMeter, get_runtime_metrics


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

    # === Helper functions ===

    def get_runtime_metrics(self):
        """Get the runtime metrics."""
        metrics_runtime = get_runtime_metrics()

        def sanitize_runtime_key(key: str) -> str:
            if key.endswith("_last"):
                return f"runtime_last/{key[8:]}"
            elif key.endswith("_avg"):
                return f"runtime_avg/{key[8:]}"
            else:
                return key

        metrics_runtime = {
            sanitize_runtime_key(key): value for key, value in metrics_runtime.items()
        }
        return metrics_runtime

    def log_as_texts(
        self,
        dict_name_to_objects: Dict[str, str],
        log_subdir: str,
    ):
        """Log objects in a directory as text. For each (key, value) in the directory, the file <log_dir>/<log_subdir>/<key> will be created and the value will be written in it.
        It will do that for all <log_dir> in self.list_run_names.

        Args:
            dict_name_to_text (Dict[str, str]): a mapping from the name of the file to create to the text to write in it.
            log_subdir (str): the subdirectory in which to create the files. If None, it will be created in the root log directory of the run.
        """
        if log_subdir is None:
            log_subdir = ""

        with RuntimeMeter("log_as_texts"):
            list_log_dirs: List[str] = []
            for log_dir_global in self.list_log_dirs_global:
                list_log_dirs.append(os.path.join(log_dir_global, log_subdir))

            for log_dir in list_log_dirs:
                os.makedirs(log_dir, exist_ok=True)
                for name, obj in dict_name_to_objects.items():
                    log_file = os.path.join(log_dir, name)
                    if not isinstance(obj, str):
                        obj = json.dumps(obj, indent=4, ensure_ascii=False)
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(obj)
                    f.close()
                    print(f"[LOGGING] : Logged {log_file}")
