import random
from typing import Any, Dict, List, Tuple, Type, Union, Set
import numpy

from core.curriculums.base_curriculum import BaseCurriculum
from core.task import TaskRepresentation


class CurriculumByLevels(BaseCurriculum):
    """A curriculum model a dynamical task distribution that is updated depending on the performance of the agent.
    Initially, only very basic tasks are part of its distribution, and more complex tasks are added as the agent's performance improves.

    The CurriculumByLevels class in particular group tasks by level and only distirbute tasks (uniformly for now)
    for levels between 1 and n+1 where n is the highest level the agent has reached.
    """

    def __init__(self, levels: List[Set[TaskRepresentation]]):
        for level in levels:
            assert isinstance(
                level, set
            ), f"Levels should be sets, got {level} (type {type(level)})"
            assert len(level) > 0, f"Levels should not be empty."
        self.levels: List[Dict[TaskRepresentation, bool]] = [
            {obj: False for obj in level} for level in levels
        ]
        self.n_levels = len(levels)
        self.idx_max_level = 0

    def sample(self) -> TaskRepresentation:
        levels_pool = self.levels[: min(self.idx_max_level, self.n_levels) + 1]
        p = p=[len(level) for level in levels_pool]
        p = [p_i / sum(p) for p_i in p]
        level_sampled: Dict[TaskRepresentation, bool] = numpy.random.choice(
            levels_pool,
            p=p,
        )  # uniform distribution
        task = random.choice(list(level_sampled.keys()))
        print(f"[CURRICULUM] Sampled task: {task}")
        return task

    def update(self, task: TaskRepresentation, feedback: Dict[str, Any]):

        if self.idx_max_level == self.n_levels:
            return  # do not update if all levels have been completed

        if not feedback["success"]:
            return  # do not update if the task was not successful

        for idx_level, level in enumerate(self.levels):
            if task in level:  # identify the task level
                if idx_level != self.idx_max_level:
                    return  # if the task is in an already solved level, do not update
                else:
                    level[task] = True  # note the task as completed
                    if all(level.values()):
                        # If all tasks in the level have been completed, update the max level
                        print(f"[CURRICULUM] Level {idx_level} has been completed.")
                        self.idx_max_level += 1
                        # If this was the last level, print a message
                        if self.idx_max_level == self.n_levels:
                            print(
                                f"[CURRICULUM] All levels have been completed. No more tasks to distribute."
                            )
                    return  # stop the loop here
