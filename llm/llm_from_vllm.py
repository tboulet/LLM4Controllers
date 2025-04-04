from abc import ABC, abstractmethod
import os
import subprocess
import time
from typing import Any, Dict, List, Union

import numpy as np
import requests
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
            # os.system(f"vllm serve {self.model} &")
            self.launch_vllm_server(
                n_gpu=1,
                model_name=self.model,
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

    def launch_vllm_server(
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
        if enable_prefix_caching:
            command.append("--enable-prefix-caching")

        try:
            process = subprocess.Popen(command)
            print("Server launching...")
            return process
        except Exception as e:
            print(f"Failed to launch server: {str(e)}")
            return None

    def wait_for_server2start(max_retries=60, delay=10, port=8000):
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
