from abc import ABC, abstractmethod
import os
import subprocess
import time
from typing import Any, Dict, List, Union

import numpy as np
import requests
import tiktoken
from tbutils.tmeasure import RuntimeMeter
from tbutils.exec_max_n import print_once

from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI

from llm.utils import (
    get_model_memory_from_model_name,
    get_model_memory_from_params,
    get_memory_allocated,
    get_memory_reserved,
    get_GPUtil_metrics,
)


class LLM_from_VLLM(LanguageModel):
    """A language model that starts a VLLM model on a server and generates completions."""

    def __init__(self, config: Dict[str, Any], logger: BaseLogger = NoneLogger()):
        self.model: str = config["model"]
        self.dtype_half: bool = config["dtype_half"]
        self.logger = logger
        self.config_server: Dict[str, Any] = config["config_server"]
        self.logger.log_scalars(
            {
                "inference_metrics/memory_model_from_name": get_model_memory_from_model_name(
                    self.model
                ),
                "inference_metrics/memory_model_torch_allocated_before": get_memory_allocated(),
                "inference_metrics/memory_model_torch_reserved_before": get_memory_reserved(),
                **get_GPUtil_metrics("inference_metrics/gputil/"),
            },
            step=0,
        )
        if self.config_server["do_server"]:
            print("Starting VLLM server...")
            # os.system(f"vllm serve {self.model} &")
            self.launch_vllm_server(
                model_name=self.model,
                n_gpu=1,
                model_len=5000,
                enable_prefix_caching=True,
                gpu_memory_utilization=0.9,
            )
            if not self.wait_for_server2start(
                delay=10,
                port=8000,
            ):
                raise RuntimeError("VLLM server failed to start.")
            print("VLLM server started.")
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
        self.kwargs: Dict[str, Any] = config.get("kwargs", {})
        self.messages: List[Dict[str, str]] = []
        self.n_tokens_in_messages = 0
        self.n_chars_in_messages = 0
        self.logger.log_scalars(
            {
                "inference_metrics/memory_model_torch_allocated": get_memory_allocated(),
                "inference_metrics/memory_model_torch_reserved": get_memory_reserved(),
                **get_GPUtil_metrics("inference_metrics/gputil/"),
            },
            step=0,
        )

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
                "inference_metrics/n_tokens_in_answer": 66,
                "inference_metrics/memory_model_torch_allocated": get_memory_allocated(),
                "inference_metrics/memory_model_torch_reserved": get_memory_reserved(),
                **get_GPUtil_metrics("inference_metrics/gputil/"),
            }
        )
        # Warn if the call finished because of too long answer
        for c in choices:
            if c.finish_reason == "length":
                print(
                    (
                        f"[WARNING] The answer was cut because it was too long.\n"
                        f"n_tokens_in_messages: {self.n_tokens_in_messages}\n"
                        f"n_tokens_in_answer: {66}"
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
        self.messages.append({"role": "assistant", "content": answer})

    def optimize(self):
        raise NotImplementedError  # not implemented yet

    def launch_vllm_server(
        self,
        model_name,
        n_gpu,
        model_len=32000,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.97,
    ):
        command = [
            "vllm",
            "serve",
            model_name,
            "--tensor-parallel-size",
            str(n_gpu),
            "--max-model-len",
            str(model_len),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--swap-space",
            "8",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
        ]
        if self.dtype_half:
            command.append("--dtype=half")
        if enable_prefix_caching:
            command.append("--enable-prefix-caching")

        try:
            process = subprocess.Popen(command)
            print("Server launching...")
            return process
        except Exception as e:
            print(f"Failed to launch server: {str(e)}")
            return None

    def wait_for_server2start(self, max_retries=60, delay=10, port=8000):
        for _ in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    print("Server is ready.")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(delay)
        print("Server failed to start in time.")
        return False
