from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Union

import numpy as np
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI


class LLM_from_VLLM(LanguageModel):
    """A language model that starts a VLLM model on a server and generates completions."""

    def __init__(self, config: Dict[str, Any]):
        self.model: str = config["model"]
        self.config_server: Dict[str, Any] = config["config_server"]
        if self.config_server["do_server"]:
            print("Starting VLLM server...")
            os.system(f"vllm serve {self.model} &")
            print("VLLM server started.")
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
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
