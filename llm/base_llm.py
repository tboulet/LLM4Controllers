from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

class LanguageModel(ABC):
    """This abstract class define the interface for a language model.
    """
    
    @abstractmethod
    def reset(self):
        """Reset the language model at empty state.
        """
    
    @abstractmethod
    def add_prompt(self, prompt : str):
        """Add the prompt to the language model.
        
        Args:
            prompt (str): the prompt to add.
        """
        
    @abstractmethod
    def generate(self) -> str:
        """Generate a completion based on its current state.
        
        Returns:
            str: the completion of the prompt.
        """
    
    @abstractmethod
    def add_answer(self, answer : str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """
        
    def optimize(self):
        raise NotImplementedError # not implemented yet