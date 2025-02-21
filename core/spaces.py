from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union
import gymnasium

from core.utils import one_time_warning


class FiniteSpace(gymnasium.Space[object]):
    
    def __init__(self, elems: List[object]):
        assert isinstance(elems, list) and len(elems) > 0, "elems must be a non-empty list"
        self.elems = elems

    def sample(self) -> object:
        return random.choice(self.elems)

    def contains(self, x: object) -> bool:
        return x in self.elems

    def __repr__(self) -> str:
        return f"FiniteSpace({self.elems})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FiniteSpace):
            return False
        elif self.elems != other.elems:
            if set(self.elems) == set(other.elems):
                one_time_warning("Warning: elements are not equal but their sets are equal, you may want to fix the order of elements")
            return False
        else:
            return True

