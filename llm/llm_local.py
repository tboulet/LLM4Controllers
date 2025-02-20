from typing import Any, Dict, List, Union
from .base_llm import LanguageModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class LLM_local(LanguageModel):
    """A language model that load locally the model."""

    def __init__(self, config: Dict[str, Any]):
        # Device
        self.device: str = config["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            print(f"Warning: CUDA is not available, using CPU instead.")
            self.device = "cpu"
        # Model and tokenizer
        self.model_name: str = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
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
        assert len(self.message) > 0, "No prompt to generate completion."
        message_to_complete = f"{self.message}\n\nAI assistant:"
        tokens = self.tokenizer(message_to_complete, return_tensors="pt").to(self.device)
        outputs = self.model(**tokens)
        message_completed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = message_completed[len(message_to_complete):].lstrip("\n\t ")
        breakpoint()
        return answer
        
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
