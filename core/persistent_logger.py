# A logger-style class that accumulate metrics and can log them, 
# but also log 0 for all already seen metrics that are not in the current metrics.

from collections import defaultdict


class PersistentLogger:
    """
    A logger that persists metrics over time.
    It logs all metrics, even if they are not present in the current metrics (as 0 in this case).
    """

    def __init__(self):
        self.seen_metrics = set()

    def reset(self):
        """
        Reset the logger.
        """
        self.metrics = {k : 0 for k in self.seen_metrics}

    def add(self, name : str, value : float):
        """
        Add a metric to the logger.

        Args:
            name (str): The name of the metric.
            value (float): The value of the metric.
        """
        if name not in self.metrics:
            self.metrics[name] = 0
        self.metrics[name] += value
        
    def log():
        defaultdict()