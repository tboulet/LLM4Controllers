
from typing import Dict, Type

from llm.llm_from_hf import LLM_from_HuggingFace
from llm.llm_from_vllm import LLM_from_VLLM
from .base_llm import LanguageModel
from .llm_from_api import LLM_from_API

llm_name_to_LLMClass : Dict[str, Type[LanguageModel]] = {
    "FromAPI" : LLM_from_API,
    "HuggingFace" : LLM_from_HuggingFace,
    "VLLM" : LLM_from_VLLM,
}