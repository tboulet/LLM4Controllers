from typing import Dict, Type
from agent.base_agent import BaseAgent
from agent.human import HumanAgent
from agent.random import RandomAgent
from agent.llm_hcg import LLMBasedHierarchicalControllerGenerator


agent_name_to_AgentClass: Dict[str, Type[BaseAgent]] = {
    "Human": HumanAgent,
    "Random": RandomAgent,
    "LLM-HCG": LLMBasedHierarchicalControllerGenerator,
}
