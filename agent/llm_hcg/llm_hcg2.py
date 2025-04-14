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
from agent.llm_hcg.llm_hcg import HCG
from core.feedback_aggregator import FeedbackAggregated
from core.task import Task, TaskDescription
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union
from hydra.utils import instantiate
from llm import llm_name_to_LLMClass


class HCG_2(HCG):
    """A temp version of HCG where the update is improved."""
    
    def update(
        self,
        task: Task,
        task_repr: TaskDescription,
        controller: Controller,
        feedback: FeedbackAggregated,
    ):
        # Add the transition to the demo bank
        self.demo_bank.add_transition(
            transition=TransitionData(
                task=task, task_repr=task_repr, code=self.sc_code_last, feedback=feedback
            )
        )
        # Log the feedback and the task description
        self.log_texts(
            {
                "feedback.txt": json.dumps(feedback.get_repr(), indent=4),
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
        examples_demobank = []
        for idx_task, transition_data in enumerate(transitions_sampled):
            examples_demobank.append(
                (
                    f"Task no {idx_task+1} :\n"
                    f"{transition_data.task_repr}\n\n"
                    f"Specialized controller code :\n```python\n{transition_data.code}\n```\n\n"
                    f"Performance : \n{transition_data.feedback}"
                )
            )

        examples_demobank = "\n\n".join(
            str(transition) for transition in examples_demobank
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
            f"{self.text_example_pc}```"
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
            f"{self.text_example_sc2}\n"
            "===========================\n\n"
            "If the task is more complicated, new, or requires some combinations of primitive controllers, you can define a new controller class "
            "before instanciating it as in the example below. But in that case, you MUST name it as 'SpecializedController' and make sure it inherits from the class Controller. \n"
            "=== Example 2 of refactoring ===\n"
            "Refactored controller for task 3:\n"
            f"{self.text_example_sc2}\n"
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
                assert len(code_primitives) <= self.n_max_update_new_primitives, (
                    f"You can only add up to {self.n_max_update_new_primitives} new controllers to the library. "
                    f"Here, you tried to add {len(code_primitives)}."
                )
                assert len(dict_code_refactorings) <= self.n_max_update_refactorings, (
                    f"You can only refactor up to {self.n_max_update_refactorings} controllers from the demo bank. "
                    f"Here, you tried to refactor {len(dict_code_refactorings)}."
                )
                assert all(
                    [
                        idx_task in range(self.n_samples_update)
                        for idx_task in dict_code_refactorings.keys()
                    ]
                ), (
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
                    },
                    in_task_folder=True,
                )
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
                            f"failing_pc_code_saving_{i}_attempt_{no_attempt}_controller.py": code_pc,
                            f"failure_pc_code_saving_{i}_attempt_{no_attempt}_error.txt": full_error_info,
                        },
                        is_update_step=True,
                    )
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
                            f"failing_sc_code_execution_{idx_task}_attempt_{no_attempt}_controller.py": code_sc,
                            f"failure_sc_code_execution_{idx_task}_attempt_{no_attempt}_error.txt": full_error_info,
                        },
                        is_update_step=True,
                    )
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

        # Update the visualizer
        self.visualizer.update_vis()

        # Increment the time step
        self.t += 1

 