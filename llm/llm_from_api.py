from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI


class LLM_from_API(LanguageModel):
    """A language model that uses an API to generate completions."""

    def __init__(self, config: Dict[str, Any]):
        self.client: OpenAI = instantiate(config["client"])
        self.model: str = config["model"]
        self.kwargs: Dict[str, Any] = config["kwargs"]
        self.messages = []

    def reset(self):
        """Reset the language model at empty state."""
        self.messages = []

    def add_prompt(self, prompt):
        # We use user to avoid Claude incompatibility with system
        self.messages.append({"role": "user", "content": prompt})
        
    def generate(self) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt (str): the prompt to complete.

        Returns:
            str: the completion of the prompt.
        """
        answer =  self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        ).choices[0].message.content
        return answer
    
    def add_answer(self, answer: str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """
        self.messages.append({"role": "assistant", "content": answer})
        
    def optimize(self):
        raise NotImplementedError  # not implemented yet
