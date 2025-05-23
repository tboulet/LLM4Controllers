from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import GPUtil
import pynvml
from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from llm.utils import get_memory_allocated, get_memory_reserved, get_model_memory_from_model_name
from transformers import PreTrainedModel

class LanguageModel(ABC):
    """This abstract class define the interface for a language model."""

    def __init__(self, config: Dict[str, Any], logger: BaseLogger = NoneLogger()):
        """Initialize the language model.

        Args:
            config (Dict[str, Any]): the configuration of the language model.
        """
        self.config = config
        self.logger = logger
        
    @abstractmethod
    def reset(self):
        """Reset the language model at empty state."""

    @abstractmethod
    def add_prompt(self, prompt: str):
        """Add the prompt to the language model.

        Args:
            prompt (str): the prompt to add.
        """

    @abstractmethod
    def generate(self, n : int) -> List[str]:
        """Generate a completion based on its current state.
        
        Args:
            n (int): the number of completions to generate.
            
        Returns:
            List[str]: the completion of the prompt.
        """

    @abstractmethod
    def add_answer(self, answer: str):
        """Add the answer to the language model.

        Args:
            answer (str): the answer to add.
        """

    def optimize(self):
        raise NotImplementedError  # not implemented yet

    # ==== Helper methods ====

    def get_gpu_usage_info(
        self,
        model_hf: PreTrainedModel = None,
        model_name_hf: str = None,
    ) -> str:
        """Get the GPU usage information.

        Args:
            model_name_hf (str): the Hugging Face model name if any.
            
        Returns:
            str: the GPU usage information.
        """
        list_info = []
        # HF parameters method
        if model_hf is not None:
            list_info.append(f"(HF) Model num_parameters: {model_hf.num_parameters()}")
            list_info.append(f"(HF) Model num_paramters trainable: {model_hf.num_parameters(only_trainable=True)}")
        if model_name_hf is not None:
            list_info.append(f"(HF) Model Memory: {get_model_memory_from_model_name(model_name_hf):.2f} GB")
        # Torch methods for memory usage
        list_info.append(f"(Torch) Memory Allocated: {get_memory_allocated():.2f} GB")
        list_info.append(f"(Torch) Memory Reserved: {get_memory_reserved():.2f} GB")
        # PyNVML
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        list_info.append(f"(PyNVML) GPU 0 Memory Used: {info.used / 1024**3:.2f} GB")
        # GPUtil
        gpu_0_info = GPUtil.getGPUs()[0]
        list_info.append(f"(GPUtil) GPU 0 Memory Used: {gpu_0_info.memoryUsed / 1024:.2f} GB")
        list_info.append(f"(GPUtil) GPU 0 Memory Free: {gpu_0_info.memoryFree / 1024:.2f} GB")
        list_info.append(f"(GPUtil) GPU 0 Memory Total: {gpu_0_info.memoryTotal / 1024:.2f} GB")
        list_info.append(f"(GPUtil) GPU 0 Memory Util: {gpu_0_info.memoryUtil * 100:.2f}%")
        # Return
        return "\t" + "\n\t".join(list_info)