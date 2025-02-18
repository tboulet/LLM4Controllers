
from typing import Dict, Type
from .base_llm import LanguageModel
from .llm_from_api import LLM_from_API

llm_name_to_LLMClass : Dict[str, Type[LanguageModel]] = {
    "FromAPI" : LLM_from_API,
}