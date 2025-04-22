from collections import defaultdict
import json
import os
import re
import shutil
from typing import Dict, List, Optional
import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from agent.llm_hcg.graph_viz import ControllerVisualizer
from agent.llm_hcg.library_controller import ControllerLibrary
from agent.llm_hcg.demo_bank import DemoBank, TransitionData
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from core.task import Task, TaskDescription
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union
from hydra.utils import instantiate
from llm import llm_name_to_LLMClass


class HCG(BaseAgent):

    def __init__(self, config: Dict, logger: BaseLogger = None):
        super().__init__(config, logger)
        # Initialize logging
        config_logs = self.config["config_logs"]
        log_dir = config_logs["log_dir"]
        self.list_log_dirs_global: List[str] = []
        if config_logs["do_log_on_new"]:
            self.list_log_dirs_global.append(os.path.join(log_dir, config["run_name"]))
        if config_logs["do_log_on_last"]:
            self.list_log_dirs_global.append(os.path.join(log_dir, "last"))
        self.metrics_memory_aware = defaultdict(int)
        # Log the config
        self.log_texts(
            {"config.yaml": json.dumps(self.config, indent=4)}, in_task_folder=False
        )
        # Initialize LLM
        name_llm = config["llm"]["name"]
        config_llm = config["llm"]["config"]
        self.llm = llm_name_to_LLMClass[name_llm](config_llm)
        # Define the text templates
        self.prompt_system = (
            "We are in the context of solving a task-based environment. The tasks consist of interacting with the environment for several steps in order to achieve a specified goal. "
            "The objects that operates in the environment are called controllers and are sub-classes of the class Controller. "
            "We maintain a library of (primitive) controllers that are relatively low-level controllers that can be composed by the actually deployed (specialized) controllers. "
            "We also maintain a demo bank of examples (task description, code, feedback) that happened during the agent's training. "
        )
        self.text_base_controller = open("agent/base_controller.py").read()
        self.text_example_answer_inference1 = open(
            "agent/llm_hcg/example_answer_inference1.txt"
        ).read()
        self.text_example_answer_inference2 = open(
            "agent/llm_hcg/example_answer_inference2.txt"
        ).read()
        self.text_example_answer_update = open(
            "agent/llm_hcg/example_answer_update.txt"
        ).read()
        self.text_example_pc = open("agent/llm_hcg/initial_PCs/move_forward.py").read()
        self.text_example_sc1 = (
            open("agent/llm_hcg/initial_SCs/go_forward.py")
            .read()
            .replace("..utils_PCs", "controller_library")
        )
        self.text_example_sc2 = (
            open("agent/llm_hcg/initial_SCs/go_north.py")
            .read()
            .replace("..utils_PCs", "controller_library")
        )
        # Extract the inference parameters
        self.n_samples_inference = self.config.get("n_samples_inference", 5)
        self.method_inference_sampling = self.config.get(
            "method_inference_sampling", "uniform"
        )
        self.num_attempts_inference = config.get("num_attempts_inference", 5)
        # Extract the update parameters
        self.n_samples_update = self.config.get("n_samples_update", 20)
        self.method_update_sampling = self.config.get(
            "method_update_sampling", "uniform"
        )
        self.n_max_update_new_primitives = self.config.get(
            "n_max_update_new_primitives", 5
        )
        self.n_max_update_refactorings = self.config.get("n_max_update_refactorings", 3)
        self.num_attempts_update_code_extraction = self.config.get(
            "num_attempts_update_code_extraction", 5
        )
        self.num_attempts_update_pc_code_saving = self.config.get(
            "num_attempts_update_pc_code_saving", 5
        )
        self.num_attempts_update_sc_code_execution = self.config.get(
            "num_attempts_update_sc_code_execution", 5
        )
        # Initialize agent's variables
        self.t = 0  # Time step
        self.base_scope = {}  # Define the base scope that can be used by the assistant
        exec(open("agent/llm_hcg/base_scope.py").read(), self.base_scope)
        self.sc_code_last: Optional[str] = None
        # Initialize HCG visualizer
        self.visualizer = ControllerVisualizer(
            agent=self,
            log_dirs=self.list_log_dirs_global,
            **config["config_visualizer"],
        )
        # Initialize knowledge base and demo bank
        self.library_controller = ControllerLibrary(
            config_agent=config,
            visualizer=self.visualizer,
        )
        self.demo_bank = DemoBank(
            config_agent=config,
            visualizer=self.visualizer,
        )
        # Update the visualizer
        self.visualizer.update_vis()

    def give_textual_description(self, description: str):
        self.description_env = description

    def get_controller(self, task_description: TaskDescription) -> Controller:
        """Get a controller for a given task description. Does that by :
        - taking the library of controllers and the (sampled) demo bank as context
        - asking the LLM to generate a controller for the task
        - extracting the code from the answer
        - executing the code to get the controller instance
        - repeat the LLM call if the code extraction or execution failed

        Args:
            task_description (TaskRepresentation): the task description.

        Raises:
            ValueError: if the LLM failed to generate a controller too many times in a row.

        Returns:
            Controller: the controller instance.
        """
        # Reset the metrics at 0
        for key in self.metrics_memory_aware.keys():
            self.metrics_memory_aware[key] = 0

        # Extract demo bank examples to be given as in-context examples
        transitions_datas_sampled = self.demo_bank.sample_transitions(
            n_transitions=self.n_samples_inference,
            method=self.method_inference_sampling,
            task_description=task_description,
        )
        self.metrics_memory_aware["n_transitions_sampled_inference"] = len(
            transitions_datas_sampled
        )

        if len(transitions_datas_sampled) == 0:
            examples_demobank = "The demo bank is empty for now. "
        else:
            examples_demobank = "\n\n".join(
                f"Example no {idx_example+1} :\n\n{transition_data}"
                for idx_example, transition_data in enumerate(transitions_datas_sampled)
            )

        # Create the prompt for the assistant
        prompt_inference = (
            # Purpose prompt
            f"{self.prompt_system}\n\n"
            "In this step, you are asked to generate a specialized controller for that task, based on the library you have access to and using "
            "the demo bank as in-context examples. "
            "\n\n"
            # Environment prompt
            "[General description of the environment]\n"
            f"{self.description_env}\n"
            "\n\n"
            # Controller structure prompt
            "[Controller interface]\n"
            "A controller obeys the following interface:\n"
            "Note : you MUST import the objects Controller, Observation, ActionType from the 'agent.base_agent' module.\n"
            "Also, you shall import any other python module you need (numpy as np, math, etc.).\n"
            "```python\n"
            f"{self.text_base_controller}```"
            "\n\n"
            # Knowledge base prompt
            "[Controller library]\n"
            "You have here controllers that are already defined in the 'controller_library' module. "
            "You are not forced to use them (you can define your own controller class and then instanciate it) but you can use them. \n"
            "If you wish to use them, you can import them in your code using the following syntax:\n"
            "```python\n"
            "from controller_library import Controller1, Controller2\n"
            "```\n"
            "If you use them, take care of the signatures of the methods of the controllers you use. \n"
            "\n"
            f"{self.library_controller}"
            "\n\n"
            # Demo bank prompt
            "[Examples]\n"
            "Here are some examples of code used to solve similar tasks in the past. "
            "For each example, you are given the task, the task description, the code used to solve it and the feedback given to the agent after the execution of the code (performance of the agent, error detected, etc.). "
            "You can see these good/bad examples (depending on feedback) as a source of inspiration to solve the task. "
            "\n\n"
            f"{examples_demobank}"
            "\n\n"
            # Advice prompt
            "[Advices]\n"
            "You can write a new controller class and/or use the controllers that are already implemented in the knowledge base. "
            "If you define a controller from scratch, you can't define functions outside the controller class, you need to define them inside the class as methods. "
            "Your code should create a 'controller' variable that will be extracted for performing in the environment\n"
            # "\n" # (commented for now)
            # "You should try as much as possible to produce controllers that are short in terms of tokens of code. "
            # "This can be done in particular by re-using the functions and controllers that are already implemented in the knowledge base and won't cost a lot of tokens. "
            "\n"
            "Please reason step-by-step and think about the best way to solve the task before answering. "
            "\n\n"
            # Example of answer prompt (in-context learning)
            "[Examples of answer]\n"
            "If a controller from the library seems already suitable for the task, you can use it directly, as in the example below:\n"
            "=== Example 1 of answer ===\n"
            f"{self.text_example_answer_inference1}\n"
            "===========================\n\n"
            "If the task is more complicated, new, or requires some combinations of primitive controllers, you can define a new controller class "
            "before instanciating it as in the example below. But in that case, you MUST name it as 'SpecializedController' and make sure it inherits from the class Controller. \n"
            "=== Example 2 of answer ===\n"
            f"{self.text_example_answer_inference2}\n"
            "===========================\n\n"
            "\n"
            # Task prompt
            "[Task to solve]\n"
            f"You will have to implement a controller (under the variable 'controller') to solve the following task : \n{task_description}."
        )

        # Log the prompt and the task description
        self.log_texts(
            {
                "prompt_inference.txt": prompt_inference,
                "demo_bank_examples.txt": examples_demobank,
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
        self.llm.add_prompt(prompt_inference)
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

    # ================ Update step ================

    def update(
        self,
        task: Task,
        task_repr: TaskDescription,
        controller: Controller,
        feedback: FeedbackAggregated,
    ):
        # Reset the metrics at 0
        for key in self.metrics_memory_aware.keys():
            self.metrics_memory_aware[key] = 0
        self.metrics_memory_aware["timestamp"] = self.t
        
        # Add the transition to the demo bank
        self.demo_bank.add_transition(
            transition=TransitionData(
                task=task,
                task_repr=task_repr,
                code=self.sc_code_last,
                feedback=feedback,
            )
        )
        # Log the feedback and the task description
        self.log_texts(
            {
                "feedback.txt": feedback.get_repr(),
            }
        )
        # Skip if not in the update step
        if self.t % 10 != 3:  # TODO : parameterize this
            self.t += 1
            return

        # Move forward the visualizer
        self.visualizer.new_step()

        # Sample transitions from the demo bank
        transitions_sampled = self.demo_bank.sample_transitions(
            n_transitions=self.n_samples_update,
            method=self.method_update_sampling,
        )
        self.metrics_memory_aware["n_transitions_sampled_update"] = len(
            transitions_sampled
        )

        examples_demobank = "\n\n".join(
            f"Task no {idx_task+1} :\n\n{transition_data}"
            for idx_task, transition_data in enumerate(transitions_sampled)
        )

        # Create the prompt for the assistant
        prompt_update = (
            # System prompt
            f"{self.prompt_system}\n\n"
            "In this step, you are asked to 1) improve the library of controllers and 2) refactor "
            "if needed the specialized controller codes that are used in the demo bank. "
            "\n\n"
            # Environment prompt
            "[General description of the environment]\n"
            f"{self.description_env}\n"
            "\n\n"
            # Controller structure prompt
            "[Controller interface]\n"
            "A controller obeys the following interface:\n"
            "Note : you MUST import the objects Controller, Observation, and ActionType from the 'agent.base_agent' module.\n"
            "Also, you shall import any other python module you need (numpy as np, math, etc.).\n"
            "```python\n"
            f"{self.text_base_controller}```"
            "\n\n"
            # Knowledge base prompt
            "[Controller library (to improve)]\n"
            "You have here the library of controllers that are already defined in the 'controller_library' module. "
            "You cannot use them in the definition of other primitive controllers of the library, but you can use them in the definition of the specialized controllers when you refactor the demo bank. "
            "If you wish to use them, you can import them in your specialized controller code using the following syntax:\n"
            "```python\n"
            "from controller_library import Controller1, Controller2\n"
            "```\n"
            "If you use them, take care of the signatures of the methods of the controllers you use. \n"
            "\n"
            f"{self.library_controller}"
            "\n\n"
            "You are asked to improve the library of controllers by making it more modular and creating usefull primitive controllers that can be used in the specialized controllers. "
            f"You can add up to {self.n_max_update_new_primitives} new controllers to the library. "
            "You CAN'T add a controller that is already present in the library. "
            "For adding a new primitive controller to the library, include 'New primitive controller' followed by "
            "the python code of the controller (between python tags) in your answer. Example:\n"
            "=== Example of new primitive controller ===\n"
            "New primitive controller:\n"
            "```python\n"
            f"{self.text_example_pc}```\n"
            "=========================\n"
            "\n\n"
            # Demo bank prompt
            "[Demo Bank (to refactor)]\n"
            "Here are some transitions (task description, code, performance) that happened during the agent's training. "
            "Based on these experiences and the controller library, you are asked to refactor the code to improve the agent's performance. "
            "The performance is a scalar between 0 and 1. If the scalar is near 1, you can hardly improve the code but you may make it more modular by using the library. "
            "\n\n"
            f"{examples_demobank}"
            "\n\n"
            "You can refactor the controller code by using the controllers from the library, or by defining new controllers that possibly compose the primitive controllers from the library. "
            f"You can refactor between 0 to {self.n_max_update_refactorings} controllers from the demo bank. "
            "For refactoring a controller from the demo bank, include 'Refactored controller for task <idx task>' followed by "
            "the definition of the controller in your answer.\n"
            "If a controller from the library seems already suitable for the task, you can use it directly, as in the example below:\n"
            "=== Example 1 of refactoring ===\n"
            "Refactored controller for task 3:\n"
            "```python\n"
            f"{self.text_example_sc2}\n"
            "```\n"
            "===========================\n\n"
            "If the task is more complicated, new, or requires some combinations of primitive controllers, you can define a new controller class "
            "before instanciating it as in the example below. But in that case, you MUST name it as 'SpecializedController' and make sure it inherits from the class Controller. \n"
            "=== Example 2 of refactoring ===\n"
            "Refactored controller for task 3:\n"
            "```python\n"
            f"{self.text_example_sc2}\n"
            "```\n"
            "===========================\n\n"
            "\n"
            "[Advices]\n"
            "- If you define a controller from scratch, you can't define functions outside the controller class, you need to define them inside the class as methods.\n"
            "- Please reason step-by-step and think about the best way to solve the task before answering.\n"
            "- You should try as much as possible to build a modular library of controllers, with low-level and general primitive controller, and high level specialized controllers that may use them.\n"
            "\n\n"
            # Example of answer prompt (in-context learning)
            "[Example of answer]\n"
            "Here is an example of acceptable answer:\n"
            "=== Example of answer ===\n"
            f"{self.text_example_answer_update}\n"
            "===========================\n\n"
        )
        self.log_texts(
            {
                "prompt_update.txt": prompt_update,
                "controller_library.py": str(self.library_controller),
                "demo_bank.py": str(self.demo_bank),
            },
            is_update_step=True,
        )
        if self.config["config_debug"]["breakpoint_update"]:
            print(f"Update step at t={self.t}. Press 'c' to continue.")
            breakpoint()

        # Call the LLM
        self.llm.reset()
        self.llm.add_prompt(prompt_update)
        # Extract the code blocks from the answer
        for no_attempt in range(self.num_attempts_update_code_extraction):
            answer = self.llm.generate()
            self.llm.add_answer(answer)
            try:
                code_primitives, dict_code_refactorings = self.extract_update_code(
                    answer
                )
                if not len(code_primitives) <= self.n_max_update_new_primitives:
                    self.metrics_memory_aware[
                        "n_failure_update_code_extraction_n_new_primitive_exceeded"
                    ] += 1
                    raise ValueError(
                        f"You can only add up to {self.n_max_update_new_primitives} new controllers to the library. "
                        f"Here, you tried to add {len(code_primitives)}."
                    )
                if not len(dict_code_refactorings) <= self.n_max_update_refactorings:
                    self.metrics_memory_aware[
                        "n_failure_update_code_extraction_n_refactoring_exceeded"
                    ] += 1
                    raise ValueError(
                        f"You can only refactor up to {self.n_max_update_refactorings} controllers from the demo bank. "
                        f"Here, you tried to refactor {len(dict_code_refactorings)}."
                    )
                if not all(
                    [
                        idx_task in range(self.n_samples_update)
                        for idx_task in dict_code_refactorings.keys()
                    ]
                ):
                    self.metrics_memory_aware[
                        "n_failure_update_code_extraction_task_index_out_of_bound"
                    ] += 1
                    raise ValueError(
                        f"Task index in refactored controllers should be in [0, {self.n_samples_update-1}]."
                        f"But here, you tried to refactor tasks with indexes {dict_code_refactorings.keys()}."
                    )
                break
            except Exception as e:
                full_error_info = get_error_info(e)
                print(
                    f"[WARNING] : Could not extract the code from the answer. Asking the assistant to try again. Full error info : {full_error_info}"
                )
                self.log_texts(
                    {
                        f"failure_code_extraction_attempt_{no_attempt}_answer.txt": answer,
                        f"failure_code_extraction_attempt_{no_attempt}_error.txt": full_error_info,
                    },
                    is_update_step=True,
                )
                self.metrics_memory_aware[
                    "n_failure_update_code_extraction"
                ] += 1
                if self.config["config_debug"][
                    "breakpoint_update_on_failure_code_extraction"
                ]:
                    print("controller code extraction failed. Press 'c' to continue.")
                    breakpoint()
                if no_attempt >= self.num_attempts_update_code_extraction - 1:
                    raise ValueError(
                        f"Could not extract the code from the answer after {self.num_attempts_update_code_extraction} attempts. Stopping the process."
                    )
                self.llm.add_prompt(
                    (
                        f"I'm sorry, extracting the code from your answer failed. Please try again and make sure the code obeys the format following that example:\n{self.text_example_answer_update}\n"
                        f"Full error info : {full_error_info}"
                    )
                )

        # Save and log the accepted answer
        answer_accepted = answer
        self.log_texts(
            {
                "assistant_answer_accepted.txt": answer_accepted,
            },
            is_update_step=True,
        )
        self.metrics_memory_aware["n_new_primitives"] = len(code_primitives)
        self.metrics_memory_aware["n_refactorings"] = len(dict_code_refactorings)

        # Warning if nothing was generated but extract_code_update did not raise an error
        if len(code_primitives) == 0 and len(dict_code_refactorings) == 0:
            print(
                f"[WARNING] : Nothing was generated from the LLM. Unexpected behavior but possible."
            )

        # Add the new primitive controllers to the library
        for i, code_pc in enumerate(code_primitives):

            # Put the LLM state to prompt + accepted answer
            self.llm.reset()
            self.llm.add_prompt(prompt_update)
            self.llm.add_answer(answer_accepted)

            # Iterate until the PC is saved. If error, log it in the message and ask the assistant to try again.
            for no_attempt in range(self.num_attempts_update_pc_code_saving):
                try:
                    self.library_controller.add_primitive_controller(code_pc)
                    break
                except Exception as e:
                    full_error_info = get_error_info(e)
                    print(
                        f"[WARNING] : Could not save the code for a new primitive controller. Full error info : {full_error_info}"
                    )
                    self.log_texts(
                        {
                            f"failure_pc_code_saving_{i}_attempt_{no_attempt}_controller.py": code_pc,
                            f"failure_pc_code_saving_{i}_attempt_{no_attempt}_error.txt": full_error_info,
                        },
                        is_update_step=True,
                    )
                    self.metrics_memory_aware[
                        "n_failure_pc_code_saving"
                    ] += 1
                    if self.config["config_debug"][
                        "breakpoint_update_on_failure_pc_code_saving"
                    ]:
                        print(
                            "controller code saving failed during update. Press 'c' to continue."
                        )
                        breakpoint()
                    if no_attempt >= self.num_attempts_update_pc_code_saving - 1:
                        break  # Abort the adding of the primitive controller if it fails too many times

                    # Ask the LLM to regenerate the code directly
                    prompt_retry_on_failure_pc_code_saving = (
                        f"Adding new controllers to the library...\n"
                        f"An error occured while trying to add the following code's controller to the library:\n"
                        f"```python\n{code_pc}\n```\n\n"
                        f"Please try again FOR THIS CONTROLLER ONLY (you can change it's class name). "
                        "IMPORTANT : Note that your answer should now be composed of CODE ONLY (not in python balises) and follow the format of the example below:\n"
                        f"{self.text_example_pc}"
                    )
                    self.log_texts(
                        {
                            f"failure_pc_code_saving_{i}_attempt_{no_attempt}_retry_prompt.txt": prompt_retry_on_failure_pc_code_saving,
                        },
                        is_update_step=True,
                    )
                    self.llm.add_prompt(prompt_retry_on_failure_pc_code_saving)
                    code_pc = self.llm.generate()
                    self.llm.add_answer(
                        code_pc
                    )  # Add the new code to the LLM state in case of a new error
                    self.log_texts(
                        {
                            f"failure_pc_code_saving_{i}_attempt_{no_attempt}_retry_answer.txt": code_pc,
                        },
                        is_update_step=True,
                    )

        # Execute the refactoring code(s)
        for idx_task, transition_data in enumerate(transitions_sampled):

            # Skip if the LLM did not refactor this task
            if idx_task not in dict_code_refactorings:
                continue

            # Put the LLM state to prompt + accepted answer
            self.llm.reset()
            self.llm.add_prompt(prompt_update)
            self.llm.add_answer(answer_accepted)

            # Iterate until the SC is executed. If error, log it in the message and ask the assistant to try again.
            code_sc = dict_code_refactorings[idx_task]
            for no_attempt in range(self.num_attempts_update_sc_code_execution):
                try:
                    controller_instance = self.exec_code_and_get_controller(code_sc)
                    transition_data.code = code_sc
                    break
                # except NoPerfGain as e: # TODO : test if the controller is metric-wise better than the previous one, if not retry
                #     pass
                except Exception as e:
                    full_error_info = get_error_info(e)
                    print(
                        f"[WARNING] : Could not execute the code for a refactored controller. Full error info : {full_error_info}"
                    )
                    self.log_texts(
                        {
                            f"failure_update_sc_code_execution_{idx_task}_attempt_{no_attempt}_controller.py": code_sc,
                            f"failure_update_sc_code_execution_{idx_task}_attempt_{no_attempt}_error.txt": full_error_info,
                        },
                        is_update_step=True,
                    )
                    self.metrics_memory_aware[
                        "n_failure_update_sc_code_execution"
                    ] += 1
                    if self.config["config_debug"][
                        "breakpoint_update_on_failure_sc_code_execution"
                    ]:
                        print(
                            "controller code execution failed during update. Press 'c' to continue."
                        )
                        breakpoint()
                    if no_attempt >= self.num_attempts_update_sc_code_execution - 1:
                        break  # Abort the refactoring of the controller if it fails too many times

                    # Ask the LLM to regenerate the answer for that task specifically
                    prompt_retry_on_failure_sc_code_execution = (
                        f"Refactoring controllers for tasks ...\n"
                        f"An error occured while trying to refactor the controller for the task no {idx_task} using the following code:\n"
                        f"```python\n{code_sc}\n```\n\n"
                        f"The error is the following : {full_error_info}\n\n"
                        f"Please try again FOR THIS CONTROLLER ONLY. "
                        f"IMPORTANT : Note that your answer should now be composed of CODE ONLY (not in python balises) and follow the format of the example below:\n"
                        f"{self.text_example_sc2}"
                    )
                    self.log_texts(
                        {
                            f"failure_sc_code_execution_{idx_task}_attempt_{no_attempt}_retry_prompt.txt": prompt_retry_on_failure_sc_code_execution,
                        },
                        is_update_step=True,
                    )
                    self.llm.add_prompt(prompt_retry_on_failure_sc_code_execution)
                    code_sc = self.llm.generate()
                    self.llm.add_answer(
                        code_sc
                    )  # Add the new code to the LLM state in case of a new error
                    self.log_texts(
                        {
                            f"failure_sc_code_execution_{idx_task}_attempt_{no_attempt}_retry_answer.txt": code_sc,
                        },
                        is_update_step=True,
                    )

        # Log the metrics
        self.logger.log_scalars(metrics=self.metrics_memory_aware, step=self.t)
        
        # Update the visualizer
        self.visualizer.update_vis()

        # Increment the time step
        self.t += 1

    # ================ Helper functions ================

    def extract_controller_code(self, answer: str) -> str:
        """From a code corresponding to an inference, extracts the controller definition and
        instantiation code.
        The answer should contain one python code block with the controller code and instanciate a Controller variable named 'controller'.

        Answer is expected to be structured as follows:
            Reasoning:
            <reasoning>

            Controller:
            ```python
            <code>
            ```

        What will be extracted is <code>.

        Args:
            answer (str): the answer from the LLM.

        Returns:
            str: the controller code.
        """
        sc_match = re.search(r"```python\n(.*?)\n```", answer, re.DOTALL)
        sc_code = sc_match.group(1).strip() if sc_match else None
        return sc_code

    def exec_code_and_get_controller(
        self,
        code: str,
        authorize_imports: bool = True,
    ) -> Controller:
        """Execute the inference code given by the LLM and return the SC instance.

        Code is expected to be structured as follows:
        ```python
        from controller_library import Controller1, Controller2 # optional, and if authorize_imports is True
        import numpy as np

        class SpecializedController(Controller1):
            ... # code for the controller

        controller = MyController()
        ```

        Because some imports may be fictional imports from the 'controller_library' module, we first extract those and execute them.

        Args:
            code (str): a string containing the code.
            The code should be structured as :
            - import statements from the (fictional) controller_library module (optional)
            - import statements from other standard modules (optional)
            - class definition (optional)
            - controller instantiation (mandatory)
            authorize_imports (bool, optional): whether to authorize imports from the controller_library module. Defaults to True.
                If False, this will raise an error in case of such imports.

        Returns:
            Controller: the controller instance.
        """
        match = re.search(r"class\s+(\w+)\s*\(", code)
        if match:
            # If there is a class definition in the code, assert that it is named 'SpecializedController'
            if match.group(1) != "SpecializedController":
                self.metrics_memory_aware[
                    "n_failure_sc_code_execution_name_is_not_SpecializedController"
                ] += 1
                raise ValueError(
                    "A class definition was found in the code but it is not named 'SpecializedController'. "
                    "If you want to define a new controller rather than only use the classes from the library, "
                    "you should name it 'SpecializedController'."
                )
            # Assert there is at most one class definition
            if len(re.findall(r"class\s+\w+\s*\(", code)) != 1:
                self.metrics_memory_aware[
                    "n_failure_sc_code_execution_more_than_one_class_definition"
                ] += 1
                raise ValueError(
                    "There is more than one class definition in the code. "
                    "You can define at most one class in the code (1 or none). "
                    "And if you define one, it should be named 'SpecializedController'."
                )

        # Extract PC class names from 'from controller_library import ...' statements
        lines = code.split("\n")
        lines_without_controller_imports: List[str] = []
        controller_classes_names: List[str] = []

        for line in lines:
            match = re.match(r"^\s*from\s+controller_library\s+import\s+(.+)$", line)
            if match:
                if not authorize_imports:
                    self.metrics_memory_aware[
                        "n_failure_sc_code_execution_unauthorized_import"
                    ] += 1
                    raise ValueError(
                        "You are not allowed to import controllers from the controller_library module when creating a new primitive controller, only when creating a specialized controller for inference/refactoring."
                    )
                controller_classes_names.extend(
                    [cls.strip() for cls in match.group(1).split(",")]
                )
            else:
                lines_without_controller_imports.append(line)
        code_without_controller_imports = "\n".join(lines_without_controller_imports)

        # Create local scope
        local_scope = {}  # Output source for the PC classes

        # Execute controller imports first
        for controller_name in controller_classes_names:
            if not (controller_name in self.library_controller.controllers):
                self.metrics_memory_aware[
                    "n_failure_sc_code_execution_controller_not_found"
                ] += 1
                raise ValueError(
                    f"Controller {controller_name} not found in the library."
                )
            controller_code = self.library_controller.controllers[controller_name].code
            try:
                exec(controller_code, self.base_scope, local_scope)
            except Exception as e:
                print(f"[ERROR] : Could not import the controller {controller_name}.")
                self.metrics_memory_aware[
                    "n_failure_sc_code_execution_controller_import_error"
                ] += 1
                raise ValueError(
                    f"An error occured while executing the code of the controller {controller_name} from the library. Maybe don't import this controller anymore. Full error info : {get_error_info(e)}"
                )

        # Execute the remaining code
        scope_with_imports = self.base_scope.copy()
        scope_with_imports.update(local_scope)
        try:
            exec(code_without_controller_imports, scope_with_imports, local_scope)
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

    def extract_update_code(self, answer: str) -> Tuple[List[str], Dict[int, str]]:
        """From a code corresponding to an update, extracts the primitive controllers and
        refactoring code snippets.

        Answer is expected to be structured as follows:
            Reasoning:
            <reasoning>

            New primitive controller: (possibly multiple)
            ```python
            <code>
            ```

            Refactored controller for task <idx>: (possibly multiple)
            ```python
            <code>
            ```

        What will be extracted is [<code>, ...] and {<idx>: <code>, ...}.

        Args:
            answer (str): the answer from the LLM.

        Returns:
            List[str]: the primitive controllers code snippets.
            Dict[int, str]: the refactoring code snippets, indexed by the task index.
        """
        try:
            code_primitives: List[str] = []
            dict_code_refactorings: Dict[int, str] = {}

            # Regex patterns
            primitive_pattern = re.compile(
                r"New primitive controller:\n```python\n(.*?)```", re.DOTALL
            )
            refactoring_pattern = re.compile(
                r"Refactored controller for task (\d+):\n```python\n(.*?)```", re.DOTALL
            )
            all_python_snippets_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)

            # Extract all Python snippets
            all_python_snippets = all_python_snippets_pattern.findall(answer)

            # Extract primitives
            code_primitives = primitive_pattern.findall(answer)

            # Extract refactorings
            for match in refactoring_pattern.findall(answer):
                task_index = int(match[0])
                code_snippet = match[1]
                dict_code_refactorings[task_index] = code_snippet
        except Exception as e:
            print("[ERROR] : Could not extract the code.")
            self.metrics_memory_aware[
                "n_failure_update_code_extraction_code_extraction_error"
            ] += 1
            raise ValueError(
                f"An error occured while extracting the code from the answer. Full error info : {get_error_info(e)}"
            )
            
        # Validate extraction
        extracted_snippet_count = len(code_primitives) + len(dict_code_refactorings)
        if extracted_snippet_count != len(all_python_snippets):
            print(
                f"[WARNING] : Mismatch between detected Python snippets ({extracted_snippet_count}) and the total number of Python snippets ({len(all_python_snippets)})."
            )
            self.metrics_memory_aware[
                "n_failure_update_code_extraction_mismatch_snippets"
            ] += 1
            raise ValueError(
                (
                    f"Mismatch between detected Python snippets ({extracted_snippet_count}) and the total number of Python snippets ({len(all_python_snippets)}). "
                    "The only tags allowed are 'New primitive controller:' and 'Refactored controller for task <idx>:' where <idx> is an integer."
                )
            )

        return code_primitives, dict_code_refactorings

    def log_texts(
        self,
        dict_name_to_text: Dict[str, str],
        in_task_folder: bool = True,
        is_update_step: bool = False,
    ):
        """Log texts in a directory. For each (key, value) in the directory, the file <log_dir>/task_<t>/<key> will contain the value.
        It will do that for all <log_dir> in self.list_run_names.

        Args:
            dict_name_to_text (Dict[str, str]): a mapping from the name of the file to create to the text to write in it.
            in_task_folder (bool, optional): whether to log the files in the task folder (if not, log in the run folder). Defaults to True.
            is_update_step (bool, optional): whether we are in an update step (if so, replace task_t by task_t_update). Defaults to False.
        """
        list_log_dirs: List[str] = []
        for log_dir_global in self.list_log_dirs_global:
            if is_update_step:
                assert (
                    in_task_folder
                ), "is_update_step should be True if in_task_folder is True."
                log_dir = os.path.join(log_dir_global, f"task_{self.t}_update")
            elif in_task_folder:
                log_dir = os.path.join(log_dir_global, f"task_{self.t}")
            else:
                log_dir = log_dir_global
            list_log_dirs.append(log_dir)

        for log_dir in list_log_dirs:
            os.makedirs(log_dir, exist_ok=True)
            for name, text in dict_name_to_text.items():
                assert isinstance(name, str) and isinstance(
                    text, str
                ), "Keys and values of dict_name_to_text should be strings."
                log_file = os.path.join(log_dir, name)
                with open(log_file, "w") as f:
                    f.write(text)
                f.close()
                print(f"[LOGGING] : Logged {log_file}")
