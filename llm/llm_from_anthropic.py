from abc import ABC, abstractmethod
from datetime import date
import datetime
import os
import time
from typing import Any, Dict, List, Optional, Union


from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from tbutils.tmeasure import RuntimeMeter

from core.utils import average
from .base_llm import LanguageModel

from llm.model_pricing import get_model_pricing

import time
import datetime
import anthropic
import numpy as np
import tiktoken
from anthropic.types import Message, Usage


class LLM_from_Anthropic(LanguageModel):
    """A language model that uses Anthropic Claude via the API."""

    def __init__(
        self,
        model: str,
        config_inference: Dict[str, Any] = {},
        max_retries: int = 5,
        logger: BaseLogger = NoneLogger(),
    ):
        super().__init__(logger)
        # Initialize client
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Config params
        self.model: str = model
        self.config_inference: Dict[str, Any] = config_inference
        self.max_retries: int = max_retries
        # Token encoding & pricing
        self.language_encoding = tiktoken.encoding_for_model("gpt-4")  # approximation
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

        if n > 1:
            print(
                "[WARNING] Claude does not support n > 1 completions; generating sequentially."
            )

        retries = 0
        outputs: List[Message] = []
        usages: List[Usage] = []

        while len(outputs) < n and retries < self.max_retries:
            try:
                with RuntimeMeter("llm_inference"):
                    response: Message = self.client.messages.create(
                        model=self.model,
                        messages=messages,
                        **self.config_inference,
                    )
                outputs.append(response)
                usages.append(response.usage)
            except anthropic.RateLimitError as e:
                time_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                time_wait = np.clip(
                    60 * 2**retries + 10 * np.random.uniform(0, 1), 10, 600
                )
                retries += 1
                if retries >= self.max_retries:
                    raise ValueError(f"Max retries reached: {e}")
                print(
                    f"[WARNING] Rate limit error at {time_date}: {e}. Retrying in {time_wait:.2f} seconds (retry {retries}/{self.max_retries})"
                )
                time.sleep(time_wait)
            except anthropic.APIStatusError as e:
                raise ValueError(f"Anthropic API error: {e}")
            except Exception as e:
                raise ValueError(f"Unexpected error during inference: {e}")

        # Extract outputs and token counts
        texts = []
        list_n_tokens_output = []
        list_n_chars_output = []
        total_input_tokens = 0
        total_output_tokens = 0

        for response, usage in zip(outputs, usages):
            text = response.content[0].text if response.content else ""
            texts.append(text)
            list_n_tokens_output.append(usage.output_tokens)
            list_n_chars_output.append(len(text))
            total_output_tokens += usage.output_tokens

        # Metrics
        metrics_inference = {
            "inference_metrics/runtime_inference": RuntimeMeter.get_last_stage_runtime(
                "llm_inference"
            ),
            "inference_metrics/n_chars_input": sum(
                len(msg["content"]) for msg in messages
            ),
            "inference_metrics/n_tokens_input": usage.input_tokens,
            "inference_metrics/n_tokens_output_sum": sum(list_n_tokens_output),
            "inference_metrics/n_tokens_output_mean": average(list_n_tokens_output),
            "inference_metrics/n_tokens_output_max": max(list_n_tokens_output),
            "inference_metrics/n_tokens_output_min": min(list_n_tokens_output),
            "inference_metrics/n_tokens_total": usage.input_tokens
            + total_output_tokens,
            "inference_metrics/n_chars_output_sum": sum(list_n_chars_output),
            "inference_metrics/n_chars_output_mean": average(list_n_chars_output),
            "inference_metrics/n_chars_output_max": max(list_n_chars_output),
            "inference_metrics/n_chars_output_min": min(list_n_chars_output),
        }

        if (
            self.price_per_1M_token_input is not None
            and self.price_per_1M_token_output is not None
        ):
            price_input = self.price_per_1M_token_input * total_input_tokens / 1_000_000
            list_price_tokens_output = [
                self.price_per_1M_token_output * n / 1_000_000
                for n in list_n_tokens_output
            ]
            metrics_inference.update(
                {
                    "inference_metrics/price_input": price_input,
                    "inference_metrics/price_output_sum": sum(list_price_tokens_output),
                    "inference_metrics/price_output_mean": average(
                        list_price_tokens_output
                    ),
                    "inference_metrics/price_output_max": max(list_price_tokens_output),
                    "inference_metrics/price_output_min": min(list_price_tokens_output),
                    "inference_metrics/price_inference": price_input
                    + sum(list_price_tokens_output),
                }
            )

        self.logger.log_scalars(metrics_inference, step=None)
        return texts
