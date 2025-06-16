import os
from typing import Any, Dict, List, Optional, Union

from core.utils import one_time_warning
from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from .base_llm import LanguageModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    GenerationMixin,
)
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
        do_resize_token_embeddings: bool = True,
        logger: BaseLogger = NoneLogger(),
    ):
        """A Hugging Face language model that loads a model from the Hugging Face Hub.

        Args:
            model (str): the name of the model to load from the Hugging Face Hub.
            device (str): the device to use for the model. Can be "cpu" or "cuda".
            do_resize_token_embeddings (bool, optional): whether to resize the token embeddings to match the tokenizer. Defaults to True.
            logger (BaseLogger, optional): the logger to use. Defaults to NoneLogger() (no logging).
        """
        super().__init__(logger)
        # Parameters
        self.model_name: str = model
        self.device = device
        if self.device == "cuda":
            assert (
                torch.cuda.is_available()
            ), "CUDA is not available. Run on CPU or fix the issue."
        hf_token = os.getenv("HF_TOKEN")
        print(
            f"[INFO] Using Hugging Face model: {self.model_name} on device: {self.device}. Model memory: {get_model_memory_from_model_name(self.model_name)} GB."
        )
        # Model and tokenizer
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        self.model: GenerationMixin = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=hf_token,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.context_window: int = self.model.config.max_position_embeddings
        # Resizing of token embeddings
        if do_resize_token_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        # Chat template
        self.do_has_chat_template = self.tokenizer.chat_template is not None
        if not self.do_has_chat_template:
            one_time_warning(
                f"Model {self.model_name} does not support chat template. Using a basic concatenation."
            )
        # Logging
        print(f"[INFO] Loaded model {self.model_name}.")
        print(
            self.get_gpu_usage_info(model_hf=self.model, model_name_hf=self.model_name)
        )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        truncation: bool = True,
        padding: bool = False,
        max_length: Optional[int] = None,
        no_grad: bool = True,
        skip_special_tokens: bool = True,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> List[str]:

        messages = self.get_messages(prompt=prompt, messages=messages)
        assert len(messages) > 0, "No prompt to generate completion."

        kwargs["num_return_sequences"] = kwargs.get("n", 1)
        if "n" in kwargs:
            del kwargs["n"]

        kwargs["do_sample"] = do_sample
        if kwargs["num_return_sequences"] > 1:
            kwargs["do_sample"] = True

        # Apply chat template for instruct models
        if self.do_has_chat_template:
            prompt_templated = self.tokenizer.apply_chat_template(
                messages, tokenize=False
            )
        else:
            prompt_templated = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )
            # TODO : add 'Assistant:' maybe

        # Tokenize and unsure it does not exceed the max length
        if max_length is not None:
            if max_length > self.context_window:
                print_once(
                    f"[WARNING] max_length ({max_length}) is greater than the model's maximum length ({self.context_window}). Clamping to the maximum length."
                )
                max_length = self.context_window

        tokens = self.tokenizer(
            prompt_templated,
            return_tensors="pt",
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        n_prompt_tokens = tokens["input_ids"].shape[1]
        if n_prompt_tokens > self.context_window:
            raise ValueError(
                f"Input length ({tokens['input_ids'].shape[1]}) exceeds the model's maximum length ({self.context_window})."
            )
        tokens = tokens.to(self.model.device)

        # Generate the completion
        if no_grad:
            with torch.no_grad():
                outputs = self.model.generate(**tokens, **kwargs)
        else:
            outputs = self.model.generate(**tokens, **kwargs)

        # Compute usage metrics
        n_completion_tokens_per_choice = [
            out.shape[0] - n_prompt_tokens for out in outputs
        ]
        n_completion_tokens = sum(n_completion_tokens_per_choice)
        n_total_tokens = n_prompt_tokens + n_completion_tokens
        usage = {
            "prompt_tokens": n_prompt_tokens,
            "completion_tokens": n_completion_tokens,
            "total_tokens": n_total_tokens,
            "completion_tokens_per_choice": n_completion_tokens_per_choice,  # custom extra field inside usage
        }

        # Decode the tokens. Outputs shape: (n, sequence_length)
        completions = []
        for out in outputs:
            completion = self.tokenizer.decode(
                out, skip_special_tokens=skip_special_tokens
            )
            # Remove prompt prefix
            answer = completion[len(prompt_templated) :].lstrip("\n\t ")
            completions.append(answer)
            
        self.logger.log_scalars
        return completions
