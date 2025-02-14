from typing import Dict, Type
from .base_agent import BaseAgent
from .human import HumanAgent
from .random import RandomAgent
from .llm_hcg import LLMBasedHierarchicalControllerGenerator


agent_name_to_AgentClass: Dict[str, Type[BaseAgent]] = {
    "Human": HumanAgent,
    "Random": RandomAgent,
    "LLM-HCG": LLMBasedHierarchicalControllerGenerator,
}
