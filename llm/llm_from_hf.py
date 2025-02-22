import os
from typing import Any, Dict, List, Union

from core.utils import one_time_warning
from .base_llm import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
import torch

class LLM_from_HuggingFace(LanguageModel):
    """A language model that load locally a HF model."""

    def __init__(self, config: Dict[str, Any]):
        # Parameters
        self.device = config["device"]
        if self.device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available. Run on CPU or fix the issue."
        self.model_name: str = config["model"]
        self.hf_token = os.getenv("HF_TOKEN")
        # Model and tokenizer
        assert self.hf_token is not None, "You need to set the HF_TOKEN environment variable as your Hugging Face token."
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model : PreTrainedModel = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device, token=self.hf_token)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Kwargs
        self.kwargs: Dict[str, Any] = config.get("kwargs", {})
        # Logging
        print(f"[INFO] Loaded model {self.model_name}.")
        print(self.get_gpu_usage_info(model_hf=self.model, model_name_hf=self.model_name))
        
    def reset(self):
        """Reset the language model at empty state."""
        self.message = ""

    def add_prompt(self, prompt: str):
        """Add a prompt to the language model.

        Args:
            prompt (str): the prompt to add.
        """
        if len(self.message) > 0:
            self.message += "\n\n"
        self.message += f"User: {prompt}"

    def generate(self) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt (str): the prompt to complete.

        Returns:
            str: the completion of the prompt.
        """
        try:
            # Build a message by adding to the prompt the beginning of the assistant part
            assert len(self.message) > 0, "No prompt to generate completion."
            message_to_complete = f"{self.message}\n\nAI assistant:"
            # Tokenize and unsure it does not exceed the max length
            tokens = self.tokenizer(message_to_complete, return_tensors="pt")
            if tokens["input_ids"].shape[1] > self.model.config.max_position_embeddings:
                raise ValueError(f"Input length ({tokens['input_ids'].shape[1]}) exceeds the model's maximum length ({self.model.config.max_position_embeddings}).")
            # Transfer to the device
            tokens = tokens.to(self.model.device)
            # Generate the completion
            outputs = self.model.generate(**tokens, **self.kwargs)
            # Decode the completion : from tensor(?) to string
            message_completed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the completion
            answer = message_completed[len(message_to_complete):].lstrip("\n\t ")
            breakpoint()
            return answer
        except Exception as e:
            breakpoint()
            raise e
        
    def add_answer(self, answer: str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """
        if len(self.message) > 0:
            self.message += "\n\n"
        self.message += f"AI assistant: {answer}"

    def optimize(self):
        raise NotImplementedError  # not implemented yet
