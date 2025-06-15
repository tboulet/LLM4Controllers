from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from GPUtil import GPU, getGPUs
import pynvml
from core.loggers.base_logger import BaseLogger
from core.loggers.none_logger import NoneLogger
from llm.utils import get_memory_allocated, get_memory_reserved, get_model_memory_from_model_name
from transformers import PreTrainedModel

class LanguageModel(ABC):
    """This abstract class define the interface for a language model."""

    def __init__(self, logger: BaseLogger = NoneLogger()):
        """Initialize the language model.

        Args:
            logger (BaseLogger, optional): the logger to use. Defaults to NoneLogger().
        """
        self.logger = logger

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        n: int = 1,
    ) -> List[str]:
        """Generate a completion based on its current state.
        
        Args:
            prompt (Optional[str], optional): the prompt to use for the completion. Defaults to None (use messages).
            messages (Optional[List[Dict[str, str]]], optional): a list of messages to use for the completion. Defaults to None (use the prompt).
            n (int, optional): the number of completions to generate. Defaults to 1.
            
        Returns:
            List[str]: the completion of the prompt.
        """
        raise NotImplementedError
    
    def optimize(self):
        raise NotImplementedError  # not implemented yet

    # ==== Helper methods ====

    def get_messages(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        # Build the messages, assert prompt xor messages is provided
        assert (prompt is not None) ^ (
            messages is not None
        ), "Either 'prompt' or 'messages' must be provided (but not both)."
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        assert len(messages) > 0, "The 'messages' list must not be empty."
        return messages
            
    def get_gpu_usage_info(
        self,
        model_hf: PreTrainedModel = None,
        model_name_hf: str = None,
    ) -> str:
        """Get the GPU usage information.

        Args:
            model_hf (PreTrainedModel, optional): the Hugging Face model if any.
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
        list_info.append(f"(PyNVML) Memory Used: {info.used / 1024**3:.2f} GB")
        # GPUtil
        gpus = getGPUs()
        if len(gpus) > 1:
            print("[WARNING] Multiple GPUs found. Using the first one.")
        if len(gpus) == 0:
            print("[WARNING] No GPUs found.")
        else:
            gpu_0_info : GPU = getGPUs()[0]
            list_info.append(f"(GPUtil) GPU 0 Memory Used: {gpu_0_info.memoryUsed / 1024:.2f} GB")
            list_info.append(f"(GPUtil) GPU 0 Memory Free: {gpu_0_info.memoryFree / 1024:.2f} GB")
            list_info.append(f"(GPUtil) GPU 0 Memory Total: {gpu_0_info.memoryTotal / 1024:.2f} GB")
            list_info.append(f"(GPUtil) GPU 0 Memory Util: {gpu_0_info.memoryUtil * 100:.2f}%")
        # Return
        return "\t" + "\n\t".join(list_info)