import os
from typing import Any, Dict, List, Optional, Union

from core.utils import one_time_warning
from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from .base_llm import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import torch
from llm.utils import (
    get_model_memory_from_model_name,
    get_memory_allocated,
    get_memory_reserved,
    get_GPUtil_metrics,
    get_model_memory_from_params,
)
from tbutils.exec_max_n import print_once


class LLM_from_HuggingFace(LanguageModel):
    """A language model that load locally a HF model."""

    def __init__(
        self,
        model: str,
        device: str,
        logger: BaseLogger = NoneLogger(),
    ):
        super().__init__(logger)
        # Parameters
        self.model_name: str = model
        self.device = device
        if self.device == "cuda":
            assert (
                torch.cuda.is_available()
            ), "CUDA is not available. Run on CPU or fix the issue."
        self.hf_token = os.getenv("HF_TOKEN")
        print(
            f"[INFO] Using Hugging Face model: {self.model_name} on device: {self.device}. Model memory: {get_model_memory_from_model_name(self.model_name)} GB."
        )
        # Model and tokenizer
        assert (
            self.hf_token is not None
        ), "You need to set the HF_TOKEN environment variable as your Hugging Face token."
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True,
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Logging
        print(f"[INFO] Loaded model {self.model_name}.")
        print(
            self.get_gpu_usage_info(model_hf=self.model, model_name_hf=self.model_name)
        )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        n: int = 1,
        no_grad: bool = True,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> List[str]:

        messages = self.get_messages(prompt=prompt, messages=messages)
        assert len(messages) > 0, "No prompt to generate completion."

        # Apply chat template for instruct models
        if self.tokenizer.chat_template is not None:
            print_once(f"[INFO] Using chat template for model {self.model_name}.")
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            print_once(
                f"[WARNING] Model {self.model_name} does not support chat template. Using a basic concatenation."
            )
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            # TODO : add 'Assistant:' maybe
            
        # Tokenize and unsure it does not exceed the max length
        tokens = self.tokenizer(prompt, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.model.config.max_position_embeddings:
            raise ValueError(
                f"Input length ({tokens['input_ids'].shape[1]}) exceeds the model's maximum length ({self.model.config.max_position_embeddings})."
            )
        # Transfer to the device
        tokens = tokens.to(self.model.device)
        # Generate the completion
        if no_grad:
            with torch.no_grad():
                outputs = self.model.generate(**tokens, **kwargs)
        else:
            outputs = self.model.generate(**tokens, **kwargs)
        # Decode the completion : from tensor(?) to string
        completion = self.tokenizer.decode(
            outputs[0], skip_special_tokens=skip_special_tokens
        )
        # Extract the completion
        answer = completion[len(prompt) :].lstrip("\n\t ")
        return answer
