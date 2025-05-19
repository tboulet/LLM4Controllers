from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import numpy as np
import tiktoken

from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from tbutils.tmeasure import RuntimeMeter
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI

from llm.utils import (
    get_model_memory_from_model_name,
    get_model_memory_from_params,
    get_memory_allocated,
    get_memory_reserved,
)



class LLM_from_API(LanguageModel):
    """A language model that uses an API to generate completions."""

    def __init__(self, config: Dict[str, Any], logger: BaseLogger = NoneLogger()):
        self.client: OpenAI = instantiate(config["client"])
        self.model: str = config["model"]
        self.logger = logger
        self.kwargs: Dict[str, Any] = config.get("kwargs", {})
        self.language_encoding = tiktoken.encoding_for_model(
            "gpt-4"
        )  # encodings are roughly similar for any LLM
        self.messages : List[Dict[str, str]] = []
        self.n_tokens_in_messages = 0
        self.n_chars_in_messages = 0

    def reset(self):
        """Reset the language model at empty state."""
        self.messages = []
        self.n_tokens_in_messages = 0
        self.n_chars_in_messages = 0

    def add_prompt(self, prompt: str):
        """Add a prompt to the language model.

        Args:
            prompt (str): the prompt to add.
        """
        # We use user to avoid Claude incompatibility with system
        self.n_chars_in_messages += len(prompt)
        self.n_tokens_in_messages += len(self.language_encoding.encode(prompt))
        self.messages.append({"role": "user", "content": prompt})

    def generate(self) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt (str): the prompt to complete.

        Returns:
            str: the completion of the prompt.
        """
        # Unsure that the prompt is not empty
        assert (
            len(self.messages) > 0
        ), "You need to add a prompt before generating completions."
        # Perform the inference
        with RuntimeMeter("llm_inference"):
            choices = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                **self.kwargs,
                seed=np.random.randint(0, 10000),
            ).choices
        answer = choices[0].message.content
        # Log inference metrics
        self.logger.log_scalars(
            {
                "inference_metrics/runtime_inference": RuntimeMeter.get_last_stage_runtime(
                    "llm_inference"
                ),
                "inference_metrics/n_chars_in_messages": self.n_chars_in_messages,
                "inference_metrics/n_tokens_in_messages": self.n_tokens_in_messages,
                "inference_metrics/n_chars_in_answer": len(answer),
                "inference_metrics/n_tokens_in_answer": len(
                    self.language_encoding.encode(answer)
                ),
                "inference_metrics/memory_model_torch_allocated": get_memory_allocated(),
                "inference_metrics/memory_model_torch_reserved": get_memory_reserved(),
            }
        )
        # Warn if the call finished because of too long answer
        for c in choices:
            if c.finish_reason == "length":
                print(
                    (
                        f"[WARNING] The answer was cut because it was too long.\n"
                        f"n_tokens_in_messages: {self.n_tokens_in_messages}\n"
                        f"n_tokens_in_answer: {len(self.language_encoding.encode(answer))}"
                    )
                )
        # Return the answer
        return answer

    def add_answer(self, answer: str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """
        self.n_chars_in_messages += len(answer)
        self.n_tokens_in_messages += len(self.language_encoding.encode(answer))
        self.messages.append({"role": "assistant", "content": answer})

    def optimize(self):
        raise NotImplementedError  # not implemented yet
