from typing import Dict, Any

MODEL_PRICING_PER_1M = {
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "o3": {"input": 10.0, "output": 40.0},
    "o4-mini": {"input": 1.1, "output": 4.4},
    # Unsure below
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
}


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """Get the model pricing for a given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        Dict[str, float]: A dictionary containing the input and output pricing for the model.
    """
    if model_name not in MODEL_PRICING_PER_1M:
        print(f"WARNING: Model '{model_name}' not found in pricing list.")
        return None, None
    return (
        MODEL_PRICING_PER_1M[model_name]["input"],
        MODEL_PRICING_PER_1M[model_name]["output"],
    )
