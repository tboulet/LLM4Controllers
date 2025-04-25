from typing import Dict, Type


from .base_agent import BaseAgent
from .base_agent2 import BaseAgent2
from .human import HumanAgent
from .random import RandomAgent, RandomAgent2
from .llm_hcg.llm_hcg import HCG
from .llm_hcg.llm_hcg2 import HCG_2

agent_name_to_AgentClass: Dict[str, Type[BaseAgent]] = {
    "Human": HumanAgent,
    "Random": RandomAgent,
    "HCG": HCG,
    "HCG_2": HCG_2,
}

agent2_name_to_AgentClass: Dict[str, Type[BaseAgent2]] = {
    "HCG_2": HCG_2,
    "Random": RandomAgent2,
}