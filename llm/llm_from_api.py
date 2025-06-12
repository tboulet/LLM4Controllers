from abc import ABC, abstractmethod
from datetime import date
import datetime
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tiktoken

from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from tbutils.tmeasure import RuntimeMeter

from core.utils import average
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
from openai.types.chat import ChatCompletion


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

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        n: int = 1,
    ) -> List[str]:

        messages = self.get_messages(prompt=prompt, messages=messages)

        # Run the inference with retries
        retries = 0
        while retries < self.max_retries:
            try:
                with RuntimeMeter("llm_inference"):
                    response: ChatCompletion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **self.config_inference,
                        n=n,
                        seed=np.random.randint(0, 10000),
                    )
                break  # Exit the loop if the request was successful
            except RateLimitError as e:
                # Handle rate limit error
                time_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if retries >= self.max_retries:
                    raise ValueError(f"Max retries reached: {e}")
                time_wait = np.clip(60 * 2**retries + 10 * np.random.uniform(0, 1), 10, 600)
                retries += 1
                print(f"[WARNING] Rate limit error happened at {time_date} : {e}. Retrying in {time_wait} seconds (retry {retries}/{self.max_retries})")
                time.sleep(time_wait)  # Exponential backoff
            except LengthFinishReasonError as e:
                # Crash if the answer is too long
                raise ValueError(f"Length finish reason error: {e}")
        choices = response.choices
        
        # Calculate inference metrics
        list_n_tokens_output = [
            len(self.language_encoding.encode(choice.message.content))
            for choice in choices
        ]
        list_n_chars_output = [
            len(choice.message.content) for choice in choices
        ]
        metrics_inference = {
            "llm_inference/runtime_inference": RuntimeMeter.get_last_stage_runtime(
                "llm_inference"
            ),
            "llm_inference/n_chars_input": sum(len(msg["content"]) for msg in messages),
            "llm_inference/n_tokens_input": response.usage.prompt_tokens,
            "llm_inference/n_tokens_output_sum": sum(list_n_tokens_output),
            "llm_inference/n_tokens_output_mean": average(list_n_tokens_output),
            "llm_inference/n_tokens_output_max": max(list_n_tokens_output),
            "llm_inference/n_tokens_output_min": min(list_n_tokens_output),
            "llm_inference/n_tokens_total": response.usage.total_tokens,
            "llm_inference/n_chars_output_sum": sum(list_n_chars_output),
            "llm_inference/n_chars_output_mean": average(list_n_chars_output),
            "llm_inference/n_chars_output_max": max(list_n_chars_output),
            "llm_inference/n_chars_output_min": min(list_n_chars_output),
        }
        if (
            self.price_per_1M_token_input is not None
            and self.price_per_1M_token_output is not None
        ):
            price_tokens_input = (
                self.price_per_1M_token_input * response.usage.prompt_tokens / 1_000_000
            )
            list_price_tokens_output = [
                self.price_per_1M_token_output * n_tokens / 1_000_000
                for n_tokens in list_n_tokens_output
            ]
            metrics_inference.update(
                {
                    "llm_inference/price_input": price_tokens_input,
                    "llm_inference/price_output_sum": sum(list_price_tokens_output),
                    "llm_inference/price_output_mean": average(
                        list_price_tokens_output
                    ),
                    "llm_inference/price_output_max": max(list_price_tokens_output),
                    "llm_inference/price_output_min": min(list_price_tokens_output),
                    "llm_inference/price_inference": price_tokens_input
                    + sum(list_price_tokens_output),
                }
            )
        self.logger.log_scalars(metrics_inference, step=None)

        # Return the answers
        answers = [choice.message.content for choice in choices]
        return answers
