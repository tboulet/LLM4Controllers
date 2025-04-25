import random
from typing import Any, Dict, List, Tuple, Type, Union, Set
import numpy

from core.curriculums.base_curriculum import BaseCurriculum, Objective
from core.feedback_aggregator import FeedbackAggregated


class CurriculumByLevels(BaseCurriculum[Objective]):
    """A curriculum model a dynamical objective distribution that is updated depending on the performance of the agent.
    Initially, only very basic objectives are part of its distribution, and more complex objectives are added as the agent's performance improves.

    The CurriculumByLevels class in particular group objectives by level and only distirbute objectives (uniformly for now)
    for levels between 1 and n+1 where n is the highest level the agent has reached.
    """

    def __init__(self, levels: List[Set[Objective]]):
        for level in levels:
            assert isinstance(
                level, set
            ), f"Levels should be sets, got {level} (type {type(level)})"
            assert len(level) > 0, f"Levels should not be empty."
        self.levels: List[Dict[Objective, bool]] = [
            {obj: False for obj in level} for level in levels
        ]
        self.n_levels = len(levels)
        self.idx_max_level = 0

    def sample(self) -> Objective:
        levels_pool = self.levels[: min(self.idx_max_level, self.n_levels) + 1]
        p = p = [len(level) for level in levels_pool]
        p = [p_i / sum(p) for p_i in p]
        level_sampled: Dict[Objective, bool] = numpy.random.choice(
            levels_pool,
            p=p,
        )  # uniform distribution
        objective = random.choice(list(level_sampled.keys()))
        print(f"\n[CURRICULUM] Sampled objective: {objective}")
        return objective

    def get_current_objectives(self) -> List[Objective]:
        """Get all the objective that are currently available to the agent.
        This corresponds to all completed levels and the current level.

        Returns:
            List[Objective]: the list of all objectives available to the agent
        """
        objectives = []
        for idx_level, level in enumerate(self.levels):
            if idx_level <= self.idx_max_level:
                objectives.extend(level.keys())
        return objectives
    
    def update(self, objective: Objective, feedback: FeedbackAggregated):

        if self.idx_max_level == self.n_levels:
            return  # do not update if all levels have been completed

        if not feedback.dict_aggregated_feedback["success"] > 0.9:
            return  # do not update if the objective was not successful

        for idx_level, level in enumerate(self.levels):
            if objective in level:  # identify the objective level
                if idx_level != self.idx_max_level:
                    return  # if the objective is in an already solved level, do not update
                else:
                    level[objective] = True  # note the objective as completed
                    if all(level.values()):
                        # If all objectives in the level have been completed, update the max level
                        print(f"[CURRICULUM] Level {idx_level} has been completed.")
                        self.idx_max_level += 1
                        # If this was the last level, print a message
                        if self.idx_max_level == self.n_levels:
                            print(
                                f"[CURRICULUM] All levels have been completed. No more objectives to distribute."
                            )
                            input("Continue?...")
                    return  # stop the loop here
