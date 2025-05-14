import csv
import os
import numpy as np
import yaml
from typing import Dict, List, Tuple, Type, Union
from core.loggers.base_logger import BaseLogger

class LoggerCSV(BaseLogger):
    def __init__(
        self,
        log_dir: str,
        timestep_key: str = "_step",
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.timestep_key = timestep_key
        # Initialize scalar logger

        self.csv_path = os.path.join(log_dir, "scalars.csv")
        self.headers = [self.timestep_key]
        self.seen_fields = set(self.headers)

        # Create empty file and write initial header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: int,
    ):
        row_dict = {self.timestep_key: step, **metrics}
        new_keys = [k for k in metrics if k not in self.seen_fields]

        if new_keys:
            # Update header and seen fields
            self.headers.extend(new_keys)
            self.seen_fields.update(new_keys)
            self._expand_csv_with_new_keys(new_keys)

        # Build full row with missing values as ""
        complete_row = {key: row_dict.get(key, "") for key in self.headers}

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(complete_row)

    def _expand_csv_with_new_keys(self, new_keys):
        # Read all rows
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

        # Update rows with NaN ("") for new keys
        updated_rows = []
        for row in existing_rows:
            for key in new_keys:
                row[key] = ""
            updated_rows.append(row)

        # Rewrite file with updated headers and rows
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(updated_rows)