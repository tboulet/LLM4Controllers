from abc import ABC, abstractmethod
import time
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
from llm.model_pricing import get_model_pricing
from openai import RateLimitError, LengthFinishReasonError


class LLM_from_API(LanguageModel):
    """A language model that uses an API to generate completions."""

    def __init__(self, config: Dict[str, Any], logger: BaseLogger = NoneLogger()):
        super().__init__(config, logger)
        # Initialize client
        self.client: OpenAI = instantiate(config["client"])
        # Initialize config parameters
        self.model: str = config["model"]
        self.config_inference: Dict[str, Any] = config.get("config_inference", {})
        self.max_retries: int = config["max_retries"]
        # Initialize other objects
        self.language_encoding = tiktoken.encoding_for_model(
            "gpt-4"
        )  # encodings are roughly similar for any LLM
        self.price_per_1M_token_input, self.price_per_1M_token_output = (
            get_model_pricing(self.model)
        )
        # Initialize variables
        self.messages: List[Dict[str, str]] = []
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

    def generate(self, n: int) -> Union[str, List[str]]:
        """Generate a completion for the given prompt.

        Args:
            prompt (str): the prompt to complete.
            n (int): the number of completions to generate.

        Returns:
            Union[str, List[str]]: the generated completion(s).
        """
        # Unsure that the prompt is not empty
        assert (
            len(self.messages) > 0
        ), "You need to add a prompt before generating completions."
        # Perform the inference
        retries = 0
        while retries < self.max_retries:
            try:
                with RuntimeMeter("llm_inference"):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        **self.config_inference,
                        n=n,
                        seed=np.random.randint(0, 10000),
                    )
                return [choice.message.content for choice in response.choices]
            except RateLimitError as e:
                # Handle rate limit error
                print(f"Rate limit error: {e}. Retrying...")
                retries += 1
                if retries >= self.max_retries:
                    raise ValueError(f"Max retries reached: {e}")
                time_wait = np.clip(2**retries + np.random.uniform(0, 1), 10, 600)
                time.sleep(time_wait)  # Exponential backoff
            except LengthFinishReasonError as e:
                # Crash if the answer is too long
                raise ValueError(f"Length finish reason error: {e}")

        choices = response.choices
        answer = choices[0].message.content
        # Log inference metrics
        n_chars_in_answer = len(answer)
        n_tokens_in_answer = len(self.language_encoding.encode(answer))
        if self.price_per_1M_token_input is not None:
            price_input = (
                self.price_per_1M_token_input * self.n_tokens_in_messages / 1_000_000
            )
            price_output = (
                self.price_per_1M_token_output * n_tokens_in_answer / 1_000_000
            )
            price_inference = price_input + price_output
            pricing_metrics = {
                "inference_metrics/price_input": price_input,
                "inference_metrics/price_output": price_output,
                "inference_metrics/price_inference": price_inference,
            }
        else:
            pricing_metrics = {}
        self.logger.log_scalars(
            {
                "inference_metrics/runtime_inference": RuntimeMeter.get_last_stage_runtime(
                    "llm_inference"
                ),
                "inference_metrics/n_chars_in_messages": self.n_chars_in_messages,  # alternative option is to use response.usage.prompt_tokens/completion_tokens/total_tokens
                "inference_metrics/n_tokens_in_messages": self.n_tokens_in_messages,
                "inference_metrics/n_chars_in_answer": n_chars_in_answer,
                "inference_metrics/n_tokens_in_answer": n_tokens_in_answer,
                "inference_metrics/memory_model_torch_allocated": get_memory_allocated(),
                "inference_metrics/memory_model_torch_reserved": get_memory_reserved(),
                **pricing_metrics,
            }
        )
        # Warn if the call finished because of too long answer
        if choices[0].finish_reason == "length":
            print(
                (
                    f"[WARNING] The answer was cut because it was too long.\n"
                    f"n_tokens_in_messages: {self.n_tokens_in_messages}\n"
                    f"n_tokens_in_answer: {len(self.language_encoding.encode(answer))}"
                )
            )
            answer += "\n\nWARNING : The answer was cut because it was too long for the model's context window."
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
