from collections import defaultdict
import os
import re
from typing import Dict, List

from gymnasium import Space
import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from agent.base_agent2 import BaseAgent2
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from core.play import play_controller_in_task
from core.task import Task, TaskDescription
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union
from llm import llm_name_to_LLMClass


class LLM_BasedControllerGenerator(BaseAgent2):
    
    def __init__(self, config, logger : BaseLogger, env : BaseMetaEnv):
        super().__init__(config, logger, env)
        self.tasks = self.env.get_current_tasks()
        self.t = 0
        self.num_attempts_inference = config["num_attempts_inference"]
        self.sc_code_last = None
        self.base_scope = {}
        exec(open("agent/llm_hcg/base_scope.py").read(), self.base_scope)
        # Initialize logging
        self.metrics_memory_aware = defaultdict(int)
        
        # Initialize LLM
        name_llm = config["llm"]["name"]
        config_llm = config["llm"]["config"]
        self.llm = llm_name_to_LLMClass[name_llm](config_llm)
        
        # Generate the prompt
        self.description_env = env.get_textual_description()
        self.text_base_controller = open("agent/llm_hcg/text_base_controller.txt").read()
        self.text_example_answer = open(
            "agent/cg/text_example_answer.txt"
        ).read()
        self.prompt_system = (
            "You are an AI agent that is used in order to solve a task-based environment through the generation of controller objects. "
            "A controller is a class that implements the method 'act(observation: Observation) -> ActionType' "
            "and that inherits from the class Controller. You will be provided with several informations in order to help you. "
            "\n\n"
            
            # Environment prompt
            "=== General description of the environment ===\n"
            f"{self.description_env}"
            "\n\n"
            
            # Controller structure prompt
            "=== Controller interface ===\n"
            "A controller obeys the following interface:\n"
            f"{self.code_tag(self.text_base_controller)}"
            "\n\n"
            
            # Example of answer
            "=== Example of answer ===\n"
            "Here is an example of acceptable answer:\n"
            f"{self.text_example_answer}\n"
            "==========================="
            "\n\n"
        )
        
    def step(self):
        # Get the task and generate the controller
        task = self.tasks[self.t]        
        print(f"Step {self.t}, task received:\n{task}")
        list_feedback : List[FeedbackAggregated] = []
        controller = self.generate_controller(task)
        # Play the controller in the task
        feedback = play_controller_in_task(controller, task, n_episodes=10, is_eval=False)
        feedback.aggregate()
        # Log the feedback
        print(f"Feedback obtained:\n{feedback}")
        self.log_texts(
            {
                "feedback.txt": feedback.get_repr(),
            }
        )
        self.logger.log_scalars(feedback.get_metrics(task=task), step=self.t)
        self.t += 1
    
    def is_done(self):
        return self.t >= len(self.tasks)
    
    def generate_controller(self, task: Task) -> Controller:
        """Generate a controller for the given task using the LLM."""
        # Reset the metrics at 0
        for key in self.metrics_memory_aware.keys():
            self.metrics_memory_aware[key] = 0
        # Generate the prompt
        task_description = task.get_description()
        prompt_task = (
            "=== Task description ===\n"
            f"Task : {task}\n\n"
            f"Description : \n{task_description}\n\n"
            "========================\n"
            "\n\n"
            "Your answer should include a python code (inside a code block) that defines a controller class and instantiate it under a variable named 'controller'. "
        )
        prompt = self.prompt_system + prompt_task
        
        # Log the prompt and the task description
        self.log_texts(
            {
                "prompt.txt": prompt,
                "task_description.txt": str(task_description),
            },
        )

        # Breakpoint-pause at each task if the debug mode is activated
        if self.config["config_debug"]["breakpoint_inference"]:
            print(f"Task {self.t}. Press 'c' to continue.")
            breakpoint()
        
        # Iterate until the controller is generated. If error, log it in the message and ask the assistant to try again.
        is_controller_instance_generated = False
        self.llm.reset()
        self.llm.add_prompt(prompt)
        for no_attempt in range(self.num_attempts_inference):
            self.metrics_memory_aware["n_attempts_inference"] += 1
            # Ask the assistant
            answer = self.llm.generate()
            self.llm.add_answer(answer)
            # Extract the code block from the answer
            code = self.extract_controller_code(answer)
            if code is None:
                # Retry if the code could not be extracted
                print(
                    f"[WARNING] : Could not extract the code from the answer. Asking the assistant to try again. (Attempt {no_attempt+1}/{self.num_attempts_inference})"
                )
                self.llm.add_prompt(
                    "I'm sorry, extracting the code from your answer failed. Please try again and make sure the code obeys the following format:\n```python\n<your code here>\n```"
                )
                self.log_texts(
                    {f"failure_sc_code_extraction_{no_attempt}_answer.txt": answer}
                )
                self.metrics_memory_aware["n_failure_sc_code_extraction"] += 1
                if self.config["config_debug"][
                    "breakpoint_inference_on_failure_code_extraction"
                ]:
                    print("controller code extraction failed. Press 'c' to continue.")
                    breakpoint()
                continue
            # Extract the controller
            try:
                specialized_controller = self.exec_code_and_get_controller(code)
            except Exception as e:
                full_error_info = get_error_info(e)
                print(
                    f"[WARNING] : Could not execute the code from the answer. Asking the assistant to try again (Attempt {no_attempt+1}/{self.num_attempts_inference}). Full error info : {full_error_info}"
                )
                self.llm.add_prompt(
                    f"I'm sorry, an error occured while executing your code. Please try again and make sure the code is correct. Full error info : {full_error_info}"
                )
                self.log_texts(
                    {
                        f"failure_sc_code_execution_{no_attempt}_answer.txt": answer,
                        f"failure_sc_code_execution_{no_attempt}_error.txt": full_error_info,
                    }
                )
                self.metrics_memory_aware["n_failure_sc_code_execution"] += 1
                if self.config["config_debug"][
                    "breakpoint_inference_on_failure_code_execution"
                ]:
                    print("controller code execution failed. Press 'c' to continue.")
                    breakpoint()
                continue
            self.sc_code_last = code
            is_controller_instance_generated = True
            break

        self.logger.log_scalars(metrics=self.metrics_memory_aware, step=self.t)

        if is_controller_instance_generated:
            self.log_texts(
                {
                    "answer.txt": answer,
                    "specialized_controller.py": code,
                }
            )
            return specialized_controller

        else:
            raise ValueError(
                f"Could not generate a controller after {self.num_attempts_inference} attempts. Stopping the process."
            )
    
    def extract_controller_code(self, answer: str) -> str:
        sc_match = re.search(r"```python\n(.*?)\n```", answer, re.DOTALL)
        sc_code = sc_match.group(1).strip() if sc_match else None
        return sc_code
            
    def exec_code_and_get_controller(
        self,
        code: str,
    ) -> Controller:
        
        # Create local scope
        local_scope = {}  # Output source for the PC classes


        # Execute the remaining code
        scope_with_imports = self.base_scope.copy()
        try:
            exec(code, scope_with_imports, local_scope)
        except Exception as e:
            print("[ERROR] : Could not execute the code.")
            self.metrics_memory_aware[
                "n_failure_sc_code_execution_code_execution_error"
            ] += 1
            raise ValueError(
                f"An error occured while executing the code for instanciating a controller. Full error info : {get_error_info(e)}"
            )

        # Retrieve the controller instance specifically from 'controller' variable
        if "controller" in local_scope and isinstance(
            local_scope["controller"], Controller
        ):
            return local_scope["controller"]
        else:
            print("[ERROR] : Could not retrieve the controller instance.")
            self.metrics_memory_aware[
                "n_failure_sc_code_execution_controller_not_created"
            ] += 1
            raise ValueError(
                "No object named 'controller' of the class Controller found in the provided code."
            )
            
    def code_tag(self, code: str) -> str:
        """Add the code tag to the code."""
        return f"```python\n{code}\n```"