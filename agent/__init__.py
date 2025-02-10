from typing import Dict, Type
from agent.base_agent import BaseAgent
from agent.human import HumanAgent
from agent.llm_hcg import LLMBasedHierarchicalControllerGenerator


agent_name_to_AgentClass: Dict[str, Type[BaseAgent]] = {
    "LLM-HCG": LLMBasedHierarchicalControllerGenerator,
    "Human": HumanAgent,
}
