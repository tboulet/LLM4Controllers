from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union
import warnings
import gym


class FiniteSpace(gym.Space[object]):
    warned_elems_inequal_but_set_equal = False
    
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
        if self.elems != other.elems and set(self.elems) != set(other.elems) and not FiniteSpace.warned_elems_inequal_but_set_equal:
            warnings.warn("Warning: elements are not equal but their sets are equal, you may want to fix the order of elements")
            FiniteSpace.warned_elems_inequal_but_set_equal = True
            return False
        return True

