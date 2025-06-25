from collections import defaultdict
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
import os
import re
import signal
import time
from typing import Dict, Iterator, List, Optional
from tbutils.tmeasure import RuntimeMeter, get_runtime_metrics
import tiktoken

from gymnasium import Space
import numpy as np
from omegaconf import OmegaConf
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from agent.base_agent2 import BaseAgent2
from core import task
from core.types import ErrorTrace, CodeExtractionError, ControllerExecutionError
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from core.parallel import run_parallel
from core.play import play_controller_in_task
from core.task import Task, TaskDescription
from core.utils import get_error_info, sanitize_name
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union
from llm import llm_name_to_LLMClass


class AgenticLLM(BaseAgent2):

    def __init__(self, config, logger: BaseLogger, env: BaseMetaEnv):
        super().__init__(config, logger, env)
        self.log_as_texts(
            {
                "config.yaml": OmegaConf.to_yaml(config),
            },
            log_subdir="",
        )
        # Get hyperparameters
        self.max_iterations: int = config["max_iterations"]
        self.list_prompt_keys: List[str] = config["list_prompt_keys"]
        self.n_tasks_to_do: Optional[int] = config["n_tasks_to_do"]
        self.n_completions: int = config["n_completions"]
        self.n_episodes_eval: int = config["n_episodes_eval"]
        self.k_pass: int = config["k_pass"]

        # Initialize variables
        self.timestep: int = 0
        self.base_scope = {}
        exec(open("agent/llm_hcg/base_scope.py").read(), self.base_scope)

        # Initialize LLM
        print("Initializing LLM...")
        name_llm = config["llm"]["name"]
        config_llm = config["llm"]["config"]
        self.llm = llm_name_to_LLMClass[name_llm](config=config_llm, logger=logger)

        # Get tasks here
        print("Getting tasks...")
        self.tasks = self.env.get_current_tasks()[: self.n_tasks_to_do]
        self.tasks = sorted(self.tasks, key=lambda task: str(task))

        # Getting example answer / templates
        self.code_item_controller = open("agent/agentic/item_controller.py").read()
        self.code_item_knowledge = open("agent/agentic/item_knowledge.py").read()
        # self.code_item_hypothesis = open("agent/agentic/item_hypothesis.py").read()
        # self.code_item_memory_snapshot = open(
        #     "agent/agentic/item_memory_snapshot.py"
        # ).read()
        # self.code_item_test = open("agent/agentic/item_test.py").read()
        self.text_example_answer = open("agent/agentic/text_example_answer.txt").read()
        
        # === Generate the different parts of the prompt ===
        self.dict_system_prompts: Dict[str, str] = {}

        # Prompt instructions
        prompt_regarding_imports = """
Regarding imports/dependencies of items, you don't need to manually import any item in the code of an other item, they will be automatically imported.
If the import triggers a circular dependency, the code will not be executed, and you will be notified of the error.
However in case of any basic python import (e.g. `import numpy as np`) that is not already imported in the code base, you need to add it when creating a controller.
        """

        if "instructions" in self.list_prompt_keys:
            prompt_instructions = f"""You are an AI agent that must solve an unknown coding environment.
For this, you have access and you maintain a code base/knowledge base over your agentic development.

This code base takes the form of one python file where objects we will call 'items' (functions, classes, custom types, ...) are enumerated in \
an order that respect their dependencies. Each item can call other items (that are defined before it in the file), \
and can be called by other items.

These items are not all fully represented, indeed, because you are a Language Model whose context window is limited and inference time \
is costly, we save tokens by only showing the signature and the docstring of the items, and not their full implementation.
You can however visualize them fully or toggle permanently their visibility if you feel its necessary (see later : actions)

# Items :
They are different types of items that you can create (by implementing corresponding interfaces) or use (by code execution)::
- **Standard item**: the basic python function/class that does not implement any specific interface.
To create/modify a standard item, simply write a code that define the function or the class in your code-action.
To delete a standard item, write ```del item_name``` in your code-action.
To use a standard item, simply call it in your code-action.
- **Controller**: a class that implements the interface of a controller, which is used to interact with the environment. It should implement the `Controller` interface.
{self.code_tag(self.code_item_controller)}
To create/modify a controller, simply write a code that defines the class in your code-action.
To delete a controller, write ```del item_name``` in your code-action.
To use a controller, simply instantiate it in your code-action and use it as you wish.
In particular, you can use the predefined function `play_controller_in_task(controller : Controller, task_id : str, n_episodes : int = 1) -> Feedback` to play the controller in the task.
It is possible to use the controller in your code without using this function for debugging or anything you found useful to do, but it is not recommended as we don't see the interest of it.
- **Knowledge**: a class that contains information about the environment, the tasks, or the agentic framework itself.
{self.code_tag(self.code_item_knowledge)}
To create/modify a knowledge, instanciate a well-named instance of the class `Knowledge` in your code-action. E.g. `knowledge_observation_shape = Knowledge(content="The shape of the observation is (64, 64, 3).")`
To delete a knowledge, write ```del item_name``` in your code-action.


## Step-by-step process :
At each call, we ask you to answer first by eventually reason about what you should do, and then submit one or several actions (action are described later).
When you submit an action, it's result will be instantly available to you (e.g. success of a coding action, or the result of a test/information retrieval action).
Then you can submit a new action.

## Conversation refreshing mechanism :
This conversation (system prompt and succession of actions/answers) is limited by the context window of the LLM and slow inference time with increasing length.
For this reason, you will have to periodically refresh the conversation by using the `refresh` action.
This will reset the conversation by removing any traces of actions/answers, leaving only the system prompt and the code base.
Consequently, it is VERY IMPORTANT to take notes of any relevant information you obtained during the conversation, in the notes section of items.
After each answer that you consider important, you should store or update the notes of the involved items, otherwise this information will be lost for future inferences.

## Actions :
You can perform actions through code blocks between <code> and </code> tags. Actions are divided in two categories:
- **Editing actions**: these actions allow you to edit the code base by creating or modifying items. \
They will happen if you define a new function or class. To modify an object, con
To


{prompt_regarding_imports}
                """
            self.dict_system_prompts["instructions"] = prompt_instructions

        # Env prompt
        if "env" in self.list_prompt_keys:
            description_env = env.get_textual_description()
            prompt_env = (
                "## General description of the environment\n" 
                f"{description_env}"
            )
            self.dict_system_prompts["env"] = prompt_env

        # Example of answer
        if "example_answer" in self.list_prompt_keys:
            text_example_answer = open("agent/agentic/text_example_answer.txt").read()
            prompt_example_answer = (
                "## Example of answer\n"
                "Here is an example of acceptable answer:\n"
                f"{text_example_answer}"
            )
            self.dict_system_prompts["example_answer"] = prompt_example_answer

        # Code prompt
        if "code_env" in self.list_prompt_keys:
            prompt_code_env = (
                "## Code of the environment\n"
                "Here is the code of the environment. You can use it to help understand the environment :\n"
                f"{env.get_code_repr()}"
            )
            self.dict_system_prompts["code_env"] = prompt_code_env

        # TODO : add doc prompt
        
        # Initialize the prompt

    def step(self):
        with RuntimeMeter("step"):

            prompt = self.build_prompt(
                dict_system_prompts=self.dict_system_prompts,
                
            )

        # Log runtime metrics
        metrics_runtime = self.get_runtime_metrics()
        self.logger.log_scalars(metrics_runtime, step=0)

        # Move forward the iter counter
        self.timestep += 1

    def solve_task(
        self,
        task: Task,
        n_completions: int,
        n_episodes_eval: int,
        log_subdir: str,
    ) -> FeedbackAggregated:
        """Solve a task using the LLM.

        Args:
            prompt_task (str): the prompt to send to the LLM
            n_completions (int): the number of completions to perform
            n_episodes_eval (int): the number of episodes to evaluate
            log_subdir (str): the subdirectory to log the results. Results will be logged in <log_dir>/<log_subdir>/<log_name>

        Returns:
            FeedbackAggregated: the feedback over the controllers
        """
        # Generate n_completions answers for the task
        prompt_task = self.build_task_prompt(task)
        self.log_as_texts(
            {
                "prompt.txt": prompt_task,
            },
            log_subdir=log_subdir,
        )
        answers = self.llm.generate(prompt=prompt_task, n=n_completions)

        # Extract, execute, and play the controller for each answer
        feedback_over_ctrl = FeedbackAggregated(
            agg_methods=["mean", f"best@{self.k_pass}", f"worst@{self.k_pass}"]
        )
        for idx_controller, answer in enumerate(answers):
            try:
                feedback_over_eps = FeedbackAggregated()
                self.log_as_texts(
                    {
                        "answer.txt": answer,
                    },
                    log_subdir=f"{log_subdir}/completion_{idx_controller}",
                )

                # Extract the code block from the answer
                code = self.extract_controller_code(answer)
                self.log_as_texts(
                    {
                        "code.py": code,
                    },
                    log_subdir=f"{log_subdir}/completion_{idx_controller}",
                )

                # Execute the code and get the controller
                controller = self.exec_code_and_get_controller(code)

                # Play the controller in the task
                feedback_over_eps = play_controller_in_task(
                    controller,
                    task,
                    n_episodes=n_episodes_eval,
                    is_eval=False,
                    log_subdir=f"{log_subdir}/completion_{idx_controller}",
                )

            except CodeExtractionError as e:
                print(f"[WARNING] CodeExtractionError : {e}")
                feedback_over_eps.add_feedback(
                    {
                        "error": ErrorTrace(
                            f"Could not extract the code from the answer. Error : {e}"
                        ),
                    }
                )

            except ControllerExecutionError as e:
                print(f"[WARNING] ControllerExecutionError : {e}")
                feedback_over_eps.add_feedback(
                    {
                        "error": ErrorTrace(
                            f"Could not execute the code extracted from the answer. Error : {e}"
                        ),
                    }
                )

            feedback_over_eps.aggregate()
            metrics_over_eps = feedback_over_eps.get_metrics()
            feedback_over_ctrl.add_feedback(metrics_over_eps)
            self.log_as_texts(
                {
                    f"feedback_over_eps.txt": feedback_over_eps.get_repr(),
                    f"feedback_over_eps_metrics.txt": metrics_over_eps,
                },
                log_subdir=f"{log_subdir}/completion_{idx_controller}",
            )

        feedback_over_ctrl.aggregate()
        # Log the metrics
        self.log_as_texts(
            {
                f"feedback_over_ctrl.txt": feedback_over_ctrl.get_repr(),
                f"feedback_over_ctrl_metrics.txt": feedback_over_ctrl.get_metrics(),
            },
            log_subdir=log_subdir,
        )
        metrics_final = feedback_over_ctrl.get_metrics(prefix=str(task))
        self.logger.log_scalars(metrics_final, step=0)
        return feedback_over_ctrl

    def is_done(self):
        return self.timestep >= 1

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
        if len(matches) == 0:
            raise CodeExtractionError("No python code block found in the answer.")
        if len(matches) != 1:
            raise CodeExtractionError(
                f"Expected exactly one python code block in the answer, found {len(matches)}."
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
            self.metrics_storer["n_failure_sc_code_execution_code_execution_error"] += 1
            raise ControllerExecutionError(
                f"An error occured while executing the code for instanciating a controller. Full error info : {get_error_info(e)}"
            )

        # Retrieve the controller instance specifically from 'controller' variable
        if "controller" in local_scope and isinstance(
            local_scope["controller"], Controller
        ):
            return local_scope["controller"]
        else:
            self.metrics_storer[
                "n_failure_sc_code_execution_controller_not_created"
            ] += 1
            raise ControllerExecutionError(
                "No object named 'controller' of the class Controller found in the provided code."
            )

    def code_tag(self, code: str) -> str:
        """Add the python code tag to some code."""
        return f"```python\n{code}\n```"
