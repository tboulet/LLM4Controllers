import os
from typing import Any, Dict, List, Optional, Union

from core.utils import average, one_time_warning
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
from tbutils.tmeasure import RuntimeMeter


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
        add_generation_prompt: bool = True,
        skip_special_tokens: bool = True,
        no_grad: bool = True,
        do_sample: bool = False,
        return_usage: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Generate one or more completions from the model.
        This method generates completions based on the provided prompt or messages.

        Args:
            prompt (Optional[str], optional): the prompt to generate completion from. It will be interpreted as a single message from 'user'. Defaults to None (use messages).
            messages (Optional[List[Dict[str, str]]], optional): the messages to generate completion from. Each message should be a dictionary with 'role' and 'content' keys. Defaults to None (use prompt). Exactly one of prompt or messages must be provided.
            truncation (bool, optional): whether to truncate the input if it exceeds the model's maximum length (or max_length if provided). If False, it will raise an error if the input exceeds the maximum length. Defaults to True.
            padding (bool, optional): Whether to pad each completion to the same length (the longest completion, or max_length if provided). Defaults to False. This is useful for batching.
            max_length (Optional[int], optional): The maximum length of the generated completion, used in case of truncation or padding.
            add_generation_prompt (bool, optional): If True, the start of the completion (related to the assistant role) will be added to the generated text. Particularly relevant for instruct models. Defaults to True.
            skip_special_tokens (bool, optional): Whether to skip special tokens when decoding the generated completion. This is useful to avoid unwanted tokens in the output, such as <pad>, <eos>, etc. Defaults to True.
            no_grad (bool, optional): Whether to disable gradient computation during generation. This saves memory and speeds up generation, but prevents gradient tracking. Defaults to True.
            do_sample (bool, optional): Whether to sample from the model's output distribution. If False, it will use greedy decoding. If True, it will sample from the model's output distribution and allow for the use of temperature, top_k, top_p, etc. Defaults to False (greedy decoding).
            return_usage (bool, optional): Whether to return a tuple (completions, usage) where usage is a dictionnary of data about the generation. Defaults to False (only return completions).
            **kwargs (Any): Additional keyword arguments to pass to the model's generate method, such as `n`, `temperature`, `top_k`, `top_p`, etc.

        Returns:
            List[str]: A list of generated completions. The number of completions is determined by the `n` parameter in `kwargs`, or defaults to 1 if not provided.
        """

        with RuntimeMeter("tokenization"):
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
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            else:
                prompt_templated_list = [
                    f"{msg['role']}: {msg['content']}" for msg in messages
                ]
                if add_generation_prompt:
                    prompt_templated_list.append("Assistant:")
                prompt_templated = "\n".join(prompt_templated_list)

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
        with RuntimeMeter("inference"):
            if no_grad:
                with torch.no_grad():
                    outputs = self.model.generate(**tokens, **kwargs)
            else:
                outputs = self.model.generate(**tokens, **kwargs)

        # Decode the tokens. Outputs shape: (n, sequence_length)
        with RuntimeMeter("decoding"):
            completions = []
            for out in outputs:
                completion = self.tokenizer.decode(
                    out, skip_special_tokens=skip_special_tokens
                )
                # Remove prompt prefix
                answer = completion[len(prompt_templated) :].lstrip("\n\t ")
                completions.append(answer)

        # Compute usage and log metrics
        list_n_completion_tokens_per_choice = [
            out.shape[0] - n_prompt_tokens for out in outputs
        ]
        n_completion_tokens = sum(list_n_completion_tokens_per_choice)
        n_total_tokens = n_prompt_tokens + n_completion_tokens
        usage = {
            "prompt_tokens": n_prompt_tokens,
            "completion_tokens": n_completion_tokens,
            "total_tokens": n_total_tokens,
            "completion_tokens_per_choice": list_n_completion_tokens_per_choice,  # custom extra field inside usage
            "runtime_inference": RuntimeMeter.get_last_stage_runtime("inference"),
            "runtime_tokenization": RuntimeMeter.get_last_stage_runtime("tokenization"),
            "runtime_decoding": RuntimeMeter.get_last_stage_runtime("decoding"),
        }

        metrics_inference = {
            "runtime_inference": RuntimeMeter.get_last_stage_runtime("inference"),
            "runtime_tokenization": RuntimeMeter.get_last_stage_runtime("tokenization"),
            "runtime_decoding": RuntimeMeter.get_last_stage_runtime("decoding"),
            "n_tokens_input": n_prompt_tokens,
            "n_tokens_output_sum": n_completion_tokens,
            "n_tokens_output_mean": average(
                list_n_completion_tokens_per_choice
            ),
            "n_tokens_output_max": max(list_n_completion_tokens_per_choice),
            "n_tokens_output_min": min(list_n_completion_tokens_per_choice),
            "n_tokens_total": n_total_tokens,
        }
        self.logger.log_scalars(metrics_inference, step=None)

        if return_usage:
            return completions, usage
        else:
            return completions
