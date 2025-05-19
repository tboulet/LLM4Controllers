from collections import defaultdict
import os
import re
from typing import Dict, List
from tbutils.tmeasure import RuntimeMeter, get_runtime_metrics
import tiktoken

from gymnasium import Space
import numpy as np
from omegaconf import OmegaConf
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

    def __init__(self, config, logger: BaseLogger, env: BaseMetaEnv):
        super().__init__(config, logger, env)
        self.log_texts(
            {
                "config.yaml": OmegaConf.to_yaml(config),
            },
            log_dir="",
        )
        # Get tasks here
        self.tasks = self.env.get_current_tasks()
        self.tasks = sorted(self.tasks, key=lambda task: str(task))
        # Get hyperparameters
        self.num_attempts_inference = config["num_attempts_inference"]
        self.n_inferences = config["n_inferences"]
        self.n_episodes_eval = config["n_episodes_eval"]
        self.k_pass = config["k_pass"]
        # Initialize variables
        self.t = 0
        self.list_prompt_keys: List[str] = config["list_prompt_keys"]
        self.base_scope = {}
        exec(open("agent/llm_hcg/base_scope.py").read(), self.base_scope)

        # Initialize logging
        self.metrics_storer = defaultdict(int)

        # Initialize LLM
        name_llm = config["llm"]["name"]
        config_llm = config["llm"]["config"]
        self.llm = llm_name_to_LLMClass[name_llm](config=config_llm, logger=logger)
        
        # === Generate the different parts of the prompt ===

        # Prompt system
        prompt_system = (
            "You are an AI agent that is used in order to solve a task-based environment through the generation of controller objects. "
            "A controller is a class that implements the method 'act(observation: Observation) -> ActionType' "
            "and that inherits from the class Controller. You will be provided with several informations in order to help you. "
        )

        # Env prompt
        description_env = env.get_textual_description()
        prompt_env = (
            "=== General description of the environment ===\n" f"{description_env}"
        )

        # Controller structure prompt
        text_base_controller = open("agent/llm_hcg/text_base_controller.txt").read()
        prompt_controller_structure = (
            "=== Controller interface ===\n"
            "A controller obeys the following interface:\n"
            f"{self.code_tag(text_base_controller)}"
        )

        # Example of answer
        text_example_answer = open("agent/cg/text_example_answer.txt").read()
        prompt_example_answer = (
            "=== Example of answer ===\n"
            "Here is an example of acceptable answer:\n"
            f"{text_example_answer}\n"
            "==========================="
        )

        # TODO : add doc and env code prompt

        # Prompts dict
        self.prompts: Dict[str, str] = {
            "system": prompt_system,
            "env": prompt_env,
            "controller_structure": prompt_controller_structure,
            "example_answer": prompt_example_answer,
        }

    def step(self):
        with RuntimeMeter("step"):
            # Get the task
            task = self.tasks[self.t]
            print(f"Step {self.t}, task received: {task}")
            # Reset metrics
            for key in self.metrics_storer.keys():
                self.metrics_storer[key] = 0
            feedback_agg_over_controllers = FeedbackAggregated(
                agg_methods=["mean", f"best@{self.k_pass}", f"worst@{self.k_pass}"]
            )
            # Play n controllers in the task
            for i in range(self.n_inferences):
                print(f"Step {self.t}, controller {i} generation...")
                controller = self.generate_controller(
                    task, log_dir=f"task_{self.t}/controller_{i}"
                )
                # Play the controller in the task
                feedback_agg = play_controller_in_task(
                    controller,
                    task,
                    n_episodes=self.n_episodes_eval,
                    is_eval=False,
                    log_dir=f"task_{self.t}/controller_{i}",
                )
                feedback_agg.aggregate()
                self.log_texts(
                    {
                        f"feedback.txt": feedback_agg.get_repr(),
                    },
                    log_dir=f"task_{self.t}/controller_{i}",
                )
                metrics_agg_over_episodes = feedback_agg.get_metrics()
                feedback_agg_over_controllers.add_feedback(metrics_agg_over_episodes)
            feedback_agg_over_controllers.aggregate()
            # Log the metrics
            self.log_texts(
                {
                    f"feedback.txt": feedback_agg_over_controllers.get_repr(),
                },
                log_dir=f"task_{self.t}",
            )
            metrics_final = feedback_agg_over_controllers.get_metrics(prefix=str(task))
            metrics_final.update(self.metrics_storer)
            self.logger.log_scalars(metrics_final, step=self.t)
        self.logger.log_scalars(get_runtime_metrics(), step=self.t)
        # Move forward t
        self.t += 1

    def is_done(self):
        return self.t >= len(self.tasks)

    def generate_controller(self, task: Task, log_dir: str = None) -> Controller:
        """Generate a controller for the given task using the LLM."""
        # Generate the prompt
        with RuntimeMeter("env_get_description"):
            task_description = task.get_description()
        prompt_task = (
            "=== Task ===\n"
            f"{task}"
        )
        prompt_task_description = (
            "=== Task description ===\n"
            f"{task_description}"
        )
        prompt_instructions = (
            "=== Instructions ===\n"
            "Your answer should include a python code (inside a code block) that implements a class "
            "inheriting from 'Controller' and instantiate it under a variable named 'controller'. "
        )

        prompts = self.prompts.copy()
        prompts["task"] = prompt_task
        prompts["task_description"] = prompt_task_description
        prompts["instructions"] = prompt_instructions

        list_prompt = []
        for prompt_key in self.list_prompt_keys:
            assert (
                prompt_key in prompts
            ), f"Prompt key {prompt_key} not found in prompts."
            list_prompt.append(prompts[prompt_key])
        prompt = "\n\n".join(list_prompt)
        
        # Log the prompt and the task description
        self.log_texts(
            {
                "prompt.txt": prompt,
                "task_description.txt": str(task_description),
            },
            log_dir=log_dir,
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
            self.metrics_storer["n_attempts_inference"] += 1
            # Ask the assistant
            answer = self.llm.generate()
            self.llm.add_answer(answer)
            # Extract the code block from the answer
            try:
                code = self.extract_controller_code(answer)
            except Exception as e:
                # Retry if the code could not be extracted
                print(
                    f"[WARNING] : Could not extract the code from the answer. Asking the assistant to try again. (Attempt {no_attempt+1}/{self.num_attempts_inference}). Error : {e}"
                )
                self.llm.add_prompt(
                    f"I'm sorry, extracting the code from your answer failed. Please try again and make sure the code obeys the following format:\n```python\n<your code here>\n```. Error : {e}"
                )
                self.log_texts(
                    {
                        f"failure_sc_code_extraction_{no_attempt}_answer.txt": answer,
                        f"failure_sc_code_extraction_{no_attempt}_error.txt": str(e),
                    },
                    log_dir=log_dir,
                )
                self.metrics_storer["n_failure_sc_code_extraction"] += 1
                if self.config["config_debug"][
                    "breakpoint_inference_on_failure_code_extraction"
                ]:
                    print("controller code extraction failed. Press 'c' to continue.")
                    breakpoint()
                continue
            # Extract the controller
            try:
                with RuntimeMeter("exec_code"):
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
                    },
                    log_dir=log_dir,
                )
                self.metrics_storer["n_failure_sc_code_execution"] += 1
                if self.config["config_debug"][
                    "breakpoint_inference_on_failure_code_execution"
                ]:
                    print("controller code execution failed. Press 'c' to continue.")
                    breakpoint()
                continue
            is_controller_instance_generated = True
            break

        if is_controller_instance_generated:
            self.log_texts(
                {
                    "answer.txt": answer,
                    "specialized_controller.py": code,
                },
                log_dir=log_dir,
            )
            return specialized_controller

        else:
            raise ValueError(
                f"Could not generate a controller after {self.num_attempts_inference} attempts. Stopping the process."
            )

    def extract_controller_code(self, answer: str) -> str:
        """Extract a python code block from the answer.
        The answer should contain exactly one python code block, marked by the
        python code tag.

        Args:
            answer (str): the answer of the assistant

        Raises:
            ValueError: if the answer does not contain exactly one detected python code block

        Returns:
            str: the code block extracted from the answer
        """
        matches = re.findall(r"```python\n(.*?)\n```", answer, re.DOTALL)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one Python code block, found {len(matches)}."
            )
        return matches[0]

    def exec_code_and_get_controller(
        self,
        code: str,
    ) -> Controller:
        """From a python code, execute it under the base scope and return the controller instance.

        Args:
            code (str): the code of the controller

        Raises:
            ValueError: if the code is not executed correctly
            ValueError: if the code does not produce a Controller instance named 'controller'

        Returns:
            Controller: the controller instance
        """
        # Create local scope
        local_scope = self.base_scope.copy()
        try:
            exec(code, local_scope)
        except Exception as e:
            breakpoint()
            print("[ERROR] : Could not execute the code.")
            self.metrics_storer["n_failure_sc_code_execution_code_execution_error"] += 1
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
            self.metrics_storer[
                "n_failure_sc_code_execution_controller_not_created"
            ] += 1
            breakpoint()
            raise ValueError(
                "No object named 'controller' of the class Controller found in the provided code."
            )

    def code_tag(self, code: str) -> str:
        """Add the python code tag to some code."""
        return f"```python\n{code}\n```"
