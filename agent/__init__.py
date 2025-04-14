from typing import Dict, Type

from agent.llm_hcg.llm_hcg2 import HCG_2
from .base_agent import BaseAgent
from .human import HumanAgent
from .random import RandomAgent
from .llm_hcg import HCG


agent_name_to_AgentClass: Dict[str, Type[BaseAgent]] = {
    "Human": HumanAgent,
    "Random": RandomAgent,
    "HCG": HCG,
    "HCG_2": HCG_2,
}
