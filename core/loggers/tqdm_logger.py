from typing import Dict, List, Tuple, Type, Union

from tqdm import tqdm
from .base_logger import BaseLogger


class LoggerTQDM(BaseLogger):

    def __init__(self, n_total: int):
        assert isinstance(
            n_total, int
        ) and n_total >= 0, f"n_episodes should be a positive integer, got {n_total} (type {type(n_total)})"
        self.tqdm_bar = tqdm(total=n_total)
        self.t = 0
        self.tqdm_bar.set_description(f"Episode 0/{n_total}")

    def log_scalars(self, metrics, step):
        self.tqdm_bar.update(step - self.t)
        self.t = step
        self.tqdm_bar.set_description(f"Episode {step}/{self.tqdm_bar.total}")
