from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from .base_llm import LanguageModel
from hydra.utils import instantiate
from openai import OpenAI


class LLM_Dummy(LanguageModel):
    """A dummy LLM for testing purposes."""

    def __init__(self, config: Dict[str, Any], logger: BaseLogger = NoneLogger()):
        self.config = config
        self.path_answer = config["path_answer"]

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        n: int = 1,
    ) -> List[str]:

        messages = self.get_messages(prompt=prompt, messages=messages)
        # Load the answer from the input file
        if self.path_answer is None:
            file_name_input = input(f"Write answer file : inputs/")
        else:
            file_name_input = self.path_answer
        # If file_name_input == 'see', show the prompt
        if file_name_input == "see":
            print(messages)
            return self.generate()
        elif file_name_input == "break":
            breakpoint()
            return self.generate()
        else:
            while True:
                try:
                    with open(f"inputs/{file_name_input}", "r") as f:
                        answer = f.read()
                    f.close()
                    break
                except FileNotFoundError:
                    print(f"File inputs/{file_name_input} not found. Try again.")
                    file_name_input = input(f"Write answer file : inputs/")
                    continue
            return [answer] * n
