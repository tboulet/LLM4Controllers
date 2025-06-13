from typing import Dict, List, Tuple, Type, Union
from .base_logger import BaseLogger
import cProfile


class LoggerProfiler(BaseLogger):
    """Logger that profiles the code execution using cProfile."""

    def __init__(self, log_dirs: List[str] = ["logs"]):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.log_dirs = log_dirs
        print("[Profiling] Profiler started.")

    def close(self):
        self.profiler.disable()
        for log_dir in self.log_dirs:
            self.profiler.dump_stats(f"{log_dir}/profile_stats.prof")
            print(
                (
                    f"[Profiling] Profile stats dumped to {log_dir}. You can visualize the profile stats using snakeviz by running "
                    f"'snakeviz {log_dir}/profile_stats.prof'"
                )
            )
