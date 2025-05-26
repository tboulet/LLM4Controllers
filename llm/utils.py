from typing import Dict, Union
from GPUtil import getGPUs, GPU
from huggingface_hub import get_safetensors_metadata
import argparse
import sys
from tbutils.exec_max_n import print_once

import torch

# Dictionary mapping dtype strings to their byte sizes
bytes_per_dtype: Dict[str, float] = {
    "int4": 0.5,
    "int8": 1,
    "float8": 1,
    "float16": 2,
    "float32": 4,
}


def get_model_memory_from_model_name(model_id: str, dtype: str = "float16") -> Union[float, None]:
    """Get the estimated GPU memory requirement for a Hugging Face model.
    Args:
        model_id: Hugging Face model ID (e.g., "facebook/opt-350m")
        dtype: Data type for model loading ("float16", "int8", etc.)
    Returns:
        Estimated GPU memory in GB, or None if estimation fails
    Examples:
        >>> get_model_size("facebook/opt-350m")
        0.82
        >>> get_model_size("meta-llama/Llama-2-7b-hf", dtype="int8")
        6.86
    """
    try:
        if dtype not in bytes_per_dtype:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Supported types: {list(bytes_per_dtype.keys())}"
            )

        metadata = get_safetensors_metadata(model_id)
        if not metadata or not metadata.parameter_count:
            raise ValueError(f"Could not fetch metadata for model: {model_id}")

        model_parameters = list(metadata.parameter_count.values())[0]
        model_parameters = int(model_parameters) / 1_000_000_000  # Convert to billions
        return round((model_parameters * 4) / (32 / (bytes_per_dtype[dtype] * 8)) * 1.18, 2)

    except Exception as e:
        print(f"Error estimating model size: {str(e)}", file=sys.stderr)
        return 66
    
def get_model_memory_from_params(n_tokens_io, n_params, batch_size=1, n_embedding=768):
    """
    Computes the model memory, input memory, and attention memory for a transformer model.
    
    Args:
        n_tokens_io (int): Combined value of n_i + n_o (input size + number of generated tokens)
        n_params (int): Number of parameters in the model
        batch_size (int): Batch size for the computation
        n_embedding (int): Size of the embeddings (typically 768 for GPT-2-like models)   
    
    Returns:
        the total memory required by the model in GB 
    """
    # Constants
    precision_bytes = 4  # 4 bytes for each parameter (32-bit precision)
    
    # 1. Model Memory (based on number of parameters)
    model_memory = n_params * precision_bytes  # Memory taken by model parameters

    # 2. Input Memory (based on input size n_i and batch size)
    input_memory = n_tokens_io * batch_size * n_embedding * precision_bytes

    # 3. Attention Memory (scales quadratically with the sequence length)
    attention_memory = (n_tokens_io * 2) * batch_size * precision_bytes

    # Total Memory
    total_memory = model_memory + input_memory + attention_memory

    # Return the total memory in GB
    return round(total_memory / 1e9, 2)

def get_memory_allocated():
    """Get the total GPU memory allocated by PyTorch."""
    return torch.cuda.memory_allocated() / 1024**3

def get_memory_reserved():
    """Get the total GPU memory reserved by PyTorch."""
    return torch.cuda.memory_reserved() / 1024**3

def get_GPUtil_metrics(prefix: str = "") -> Dict[str, Union[str, float]]:
    """Get metrics from GPUtil"""
    gpus = getGPUs()
    if len(gpus) == 0:
        print_once("[WARNING] No GPUs found.")
        return {}
    elif len(gpus) > 1:
        print_once("[WARNING] Multiple GPUs found. Using the first one.")
    gpu_0_info : GPU = gpus[0]
    metrics = {
        "gpu_memory_used": gpu_0_info.memoryUsed,
        "gpu_memory_percent": gpu_0_info.memoryUtil,
        "gpu_load_percent": gpu_0_info.load,
        "gpu_temperature": gpu_0_info.temperature,
    }
    return {f"{prefix}{k}": v for k, v in metrics.items()}

if __name__ == "__main__":
    model_id = input("Enter the model ID: ")
    dtype = input("Enter the data type (float16, int8, etc.): (default: float16) ")
    if dtype == "":
        dtype = "float16"
    memory_GB = get_model_memory_from_model_name(model_id, dtype)
    print(f"Estimated GPU memory requirement for model {model_id}: {memory_GB} GB")