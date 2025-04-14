from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI


class LLM_Dummy(LanguageModel):
    """A dummy LLM for testing purposes."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def reset(self):
        """Reset the language model at empty state."""
        self.messages = []

    def add_prompt(self, prompt: str):
        """Add a prompt to the language model.

        Args:
            prompt (str): the prompt to add.
        """
        # We use user to avoid Claude incompatibility with system
        self.messages.append({"role": "user", "content": prompt})

    def generate(self) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt (str): the prompt to complete.

        Returns:
            str: the completion of the prompt.
        """
        assert (
            len(self.messages) > 0
        ), "You need to add a prompt before generating completions."
        # Load the answer from the input file
        file_name_input = input(f"Write answer file : inputs/")
        # If file_name_input == 'see', show the prompt
        if file_name_input == "see":
            print(self.messages)
            return self.generate()
        elif file_name_input == "break":
            breakpoint()
            return self.generate()
        else:
            with open(f"inputs/{file_name_input}", "r") as f:
                answer = f.read()
            return answer

    def add_answer(self, answer: str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """
        self.messages.append({"role": "assistant", "content": answer})