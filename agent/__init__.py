from typing import Dict, Type
from agent.base_agent import BaseAgent
from agent.llm_hcg import LLMBasedHierarchicalControllerGenerator


agent_name_to_AgentClass: Dict[str, Type[BaseAgent]] = {
    "LLMBasedHierarchicalControllerGenerator": LLMBasedHierarchicalControllerGenerator,
}
