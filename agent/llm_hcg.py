import os
import re
from typing import Dict, List

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller, Task
from env.base_meta_env import BaseMetaEnv, Observation, Action, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union



class RandomController(Controller):
    """A controller that acts randomly in the environment."""

    def act(self, observation: Observation) -> Action:
        return random.choice(
                [
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                ]
            )
        
    def has_finished(self) -> bool:
        return False


class LLMBasedHierarchicalControllerGenerator(BaseAgent):

    def __init__(self, config: Dict):
        super().__init__(config)
        # Initialize OpenAI API
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4o-mini"
        self.namespace = {}  # Dictionary to store dynamically created variables
        
    def give_textual_description(self, description: str):
        
        # Initialize messages with the description of its purpose
        self.messages = []
        text_controller_base_class = open("agent/base_controller.py").read()
        text_controller_instanciation_example = open("assets/controller_instanciation_example.py").read()
        text_agent_answer_example = open("assets/agent_answer_example.txt").read()
        self.messages.append(
            {
                "role": "system",
                "content": (
                    "You are an agent that will be asked to perform a series of tasks in an RL-like environment. "
                    "Each task correspond to a new episode in the environment. "
                    "The description of the task will be given to you at the beginning of each episode by the User. "
                    "The general description of the environment is the following:\n\n"
                    f"{description}"
                    "\n\n"
                    
                    "You will be able to add 'controllers' to your internal library and to use them to solve the tasks. You can reuse the controllers that you have already generated for similar tasks. "
                    "Formally, you will be asked two things:\n"
                    
                    "1) Eventually generate a new controller. "
                    "In this case, the class that you should subclass is the Controller class and is the following. Please respect the signature of the methods:\n"
                    "```python\n"
                    f"{text_controller_base_class}"
                    "```\n\n"
                    
                    "2) Select the controller that you think is the most appropriate for the task among your library of controllers. "
                    "For this you will call the controller class and pass some information about the task (the class __init__ method and the information you will give it need to be coherent). "
                    "This answer should look like the following example:\n\n"
                    f"{text_controller_instanciation_example}"
                    "\n\n"
                    
                    "Please reason step-by-step and think about the best way to solve the tasks before answering. "
                    "Globally, your answer should be returned following that example:\n\n"
                    f"{text_agent_answer_example}"
                )
            }
        )
        print(f"System message : {self.messages[-1]['content']}")
        self.exec_globals = {}
        
    def get_controller(self, task: Task) -> Controller:
        self.messages.append(
            {
                "role": "user",
                "content": f"Task description : {task}\nPlease select the controller that you think is the most appropriate for this task, and eventually generate a new controller."
            }
        )
        assert isinstance(task, str), "For now, the LLM-based hierarchical controller only supports string tasks."
        self.ask_for_answer()
        
    def ask_for_answer(self):
        print(f"Asking the model {self.model} for answer...")
        # Ask the assistant
        answer_assistant = (
            self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
            .choices[0]
            .message.content
        )
        print(f"Answer of assistant: {answer_assistant}")
        
        # Extract the code blocks from the answer
        controller_def, controller_inst = self.extract_code_blocks(answer_assistant)
        
        # Execute the controller code
        exec(controller_def, globals(), self.namespace)  # Define the class
        exec(controller_inst, globals(), self.namespace)  # Instantiate the object

        # Retrieve the controller instance
        controller_instance = self.namespace.get("controller")
        
        raise
    
    def extract_code_blocks(self, answer: str):
        """Extracts the controller definition and instantiation code from an LLM response."""
        
        # Regular expressions to extract the code blocks
        controller_def_pattern = r"New controller.*?```python\n(.*?)```"
        controller_inst_pattern = r"Controller instanciation.*?```python\n(.*?)```"

        controller_def_match = re.search(controller_def_pattern, answer, re.DOTALL)
        controller_inst_match = re.search(controller_inst_pattern, answer, re.DOTALL)

        controller_def = controller_def_match.group(1).strip() if controller_def_match else None
        controller_inst = controller_inst_match.group(1).strip() if controller_inst_match else None

        return controller_def, controller_inst