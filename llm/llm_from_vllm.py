from abc import ABC, abstractmethod
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
import tiktoken
from tbutils.tmeasure import RuntimeMeter
from tbutils.exec_max_n import print_once

from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from core.utils import average
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI
from openai.types.chat import ChatCompletion

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
            print("[VLLM] VLLM server started.")

        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )
        self.kwargs: Dict[str, Any] = config.get("kwargs", {})
        self.logger.log_scalars(
            {
                "inference_metrics/memory_model_torch_allocated": get_memory_allocated(),
                "inference_metrics/memory_model_torch_reserved": get_memory_reserved(),
                **get_GPUtil_metrics("inference_metrics/gputil/"),
            },
            step=0,
        )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        n: int = 1,
    ) -> List[str]:
        # Create the messages if not provided
        messages = self.get_messages(prompt=prompt, messages=messages)

        # Perform the inference
        with RuntimeMeter("llm_inference"):
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.kwargs,
                seed=np.random.randint(0, 10000),
            )
        choices = response.choices

        # Calculate inference metrics
        list_n_chars_output = [len(choice.message.content) for choice in choices]
        metrics_inference = {
            "llm_inference/runtime_inference": RuntimeMeter.get_last_stage_runtime(
                "llm_inference"
            ),
            "llm_inference/n_chars_input": sum(len(msg["content"]) for msg in messages),
            "llm_inference/n_tokens_input": response.usage.prompt_tokens,
            "llm_inference/n_tokens_output_sum": response.usage.completion_tokens,
            "llm_inference/n_tokens_total": response.usage.total_tokens,
            "llm_inference/n_chars_output_sum": sum(list_n_chars_output),
            "llm_inference/n_chars_output_mean": average(list_n_chars_output),
            "llm_inference/n_chars_output_max": max(list_n_chars_output),
            "llm_inference/n_chars_output_min": min(list_n_chars_output),
            "inference_metrics/memory_model_torch_allocated": get_memory_allocated(),
            "inference_metrics/memory_model_torch_reserved": get_memory_reserved(),
            **get_GPUtil_metrics("inference_metrics/gputil/"),
        }
        self.logger.log_scalars(metrics_inference, step=None)

        # Warn if the call finished because of too long answer
        for c in choices:
            if c.finish_reason == "length":
                print(
                    f"[WARNING] The answer was cut because it was too long : length of {len(c.message.content)} characters. "
                )

        # Return the answers
        answers = [choice.message.content for choice in choices]
        return answers
    

    # ============= Helper methods =================

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
        if "reasoning-parser" in self.config_server:
            command.extend(
                [
                    "--reasoning-parser",
                    self.config_server["reasoning_parser"],
                ]
            )
        if "chat-template" in self.config_server:
            command.extend(
                [
                    "--chat-template",
                    self.config_server["chat-template"],
                ]
            )
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
            except requests.exceptions.RequestException as e:
                print(f"Exception occurred: {e}. Retrying in {delay} seconds...")
                pass
            time.sleep(delay)
        print("Server failed to start in time.")
        return False
