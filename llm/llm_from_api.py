from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI


class LLM_from_API(LanguageModel):
    """A language model that uses an API to generate completions."""

    def __init__(self, config: Dict[str, Any]):
        self.client: OpenAI = instantiate(config["client"])
        self.model: str = config["model"]
        self.kwargs: Dict[str, Any] = config.get("kwargs", {})
        self.messages : List[Dict[str, str]] = []

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
        answer = (
            self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                **self.kwargs,
                seed=np.random.randint(0, 10000),
            )
            .choices[0]
            .message.content
        )
        return answer

    def add_answer(self, answer: str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """
        self.messages.append({"role": "assistant", "content": answer})

    def optimize(self):
        raise NotImplementedError  # not implemented yet
