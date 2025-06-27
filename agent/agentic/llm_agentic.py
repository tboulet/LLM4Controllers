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
from typing import Dict, Iterator, List, Optional, Set
from tbutils.tmeasure import RuntimeMeter, get_runtime_metrics
import tiktoken
import io
import sys
import traceback


from gymnasium import Space
import numpy as np
from omegaconf import OmegaConf
from openai import OpenAI
from agent.agentic.codebase_manager import CodebaseManager
from agent.base_agent import BaseAgent, Controller
from agent.base_agent2 import BaseAgent2
from core import task
from core.play import play_controller_in_task
from core.std_capture import StreamCapture
from core.types import ErrorTrace, CodeExtractionError, CodeExecutionError
from core.feedback_aggregator import FeedbackAggregated
from core.loggers.base_logger import BaseLogger
from core.parallel import run_parallel
from core.task import Task, TaskDescription
from core.utils import get_error_info, sanitize_name
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union
from llm import llm_name_to_LLMClass


class TaskStatus:
    """A class to represent the status of a task in the agentic framework."""

    def __init__(
        self,
        name: str,
        task: Task,
        # controller_code: str = None,
        performance_metric: float = None,
        is_done: bool = False,
        n_submissions: int = 0,
    ):
        """Initialize the TaskStatus object.

        Args:
            name (str): the name of the task
            task (Task): the task object associated with this status.
            controller_code (str, optional): the code of the controller that solves the task. Defaults to None (no submission yet).
            performance_metric (float, optional): the performance metric of the controller on the task. Defaults to None (no submission yet).
            is_done (bool, optional): whether the task is done or not. Defaults to False (not done).
        """
        self.name: str = name
        self.task: Task = task
        # self.controller_code: str = controller_code
        self.performance_metric: float = performance_metric
        self.is_done: bool = is_done
        self.n_submissions: int = n_submissions

    def __repr__(self) -> str:
        """String representation of the TaskStatus object."""
        list_result = []
        list_result.append(f"Task name: '{self.name}'")
        if self.performance_metric is None:
            assert (
                self.is_done is False and self.n_submissions == 0
            ), "If the performance metric is None, the task should not be done and no controller should have been submitted yet."
            list_result.append("No controller submitted yet.")
        else:
            assert (
                self.n_submissions > 0
            ), "If the performance metric is not None, the number of submissions should be greater than 0."
            list_result.append(
                # f"Controller code:\n{self.controller_code}\n"
                f"Performance metric: {self.performance_metric}\n"
                f"Task done: {self.is_done}\n"
                f"Number of different controllers submitted since the beginning: {self.n_submissions}\n"
            )
        return "\n".join(list_result)


class Agentic(BaseAgent2):

    def __init__(self, config, logger: BaseLogger, env: BaseMetaEnv):
        super().__init__(config, logger, env)
        self.log_as_texts(
            {
                "config.yaml": OmegaConf.to_yaml(config),
            },
            log_subdir="",
        )
        # Get hyperparameters
        self.n_steps_max: int = config["n_steps_max"]
        self.list_prompt_keys: List[str] = config["list_prompt_keys"]
        self.n_tasks_to_do: int = config["n_tasks_to_do"]

        # Initialize variables
        self.timestep: int = 0
        self.timestep_in_conv: int = 0
        self.idx_conversation: int = (
            -1
        )  # Start with -1, will be incremented at the first step
        self.is_conversation_fresh: bool = True
        self.env_variables: Dict[str, Any] = {}
        self.time_start: float = time.time()
        self.tasks_status: Dict[str, TaskStatus] = {}

        # Initialize LLM
        print("Initializing LLM...")
        name_llm = config["llm"]["name"]
        config_llm = config["llm"]["config"]
        self.llm = llm_name_to_LLMClass[name_llm](**config_llm, logger=logger)

        # Getting example answer / templates
        self.code_item_controller = open("agent/agentic/item_controller.py").read()
        self.code_item_knowledge = open("agent/agentic/item_knowledge.py").read()
        # self.code_item_hypothesis = open("agent/agentic/item_hypothesis.py").read()
        # self.code_item_memory_snapshot = open(
        #     "agent/agentic/item_memory_snapshot.py"
        # ).read()
        # self.code_item_test = open("agent/agentic/item_test.py").read()
        self.text_example_answer = open("agent/agentic/text_example_answer.txt").read()

        # === Generate the different parts of the system prompt ===
        self.dict_system_prompts: Dict[str, str] = {}

        # Prompt instructions
        prompt_regarding_imports = """
Regarding imports/dependencies of items, you don't need to manually import any item in the code of an other item, they will be automatically imported.
If the import triggers a circular dependency, the code will not be executed, and you will be notified of the error.
However in case of any basic python import (e.g. `import numpy as np`) that is not already imported in the code base, you need to add it when creating a controller.
        """

        prompt_regarding_partial_repr = """These items are not all fully represented, indeed, because you are a Language Model whose context window is limited and inference time \
is costly, we save tokens by only showing the signature and the docstring of the items, and not their full implementation.
You can however visualize them fully or toggle permanently their visibility if you feel its necessary (see later : actions)"""

        if "instructions" in self.list_prompt_keys:
            prompt_instructions = f"""You are Agentic, an AI agent that must solve a task-based environment.
For this, you have access and you maintain a code base/knowledge base over your agentic development.

This code base takes the form of one python file where objects we will call 'items' (functions, classes, variables, ...) are enumerated in \
an order that respect their dependencies. Each item can call other items (that are defined before it in the file), \
and can be called by other items.

{prompt_regarding_partial_repr}

## Items :

They are different types of items that you can create (by code editing) or use (by code execution)::

- **Standard item**: the basic python function/class/variables that does not implement any specific interface.
To create/modify a standard item, simply write a code that define the function or class or variable in your code-edit action.
To delete a standard item, write ```del item_name``` in your code-edit action.
To use a standard item, simply call it in your code-exec action.

- **Controller**: a class that implements the interface of a controller, which is used to interact with the environment. It should implement the `Controller` interface.
{self.code_tag(self.code_item_controller)}
To create/modify a controller, simply write a code that defines the class in your code-edit action.
To delete a controller, write ```del item_name``` in your code-edit action.
To use a controller, simply instantiate it in your code-exec action and use it as you wish.
In particular, you can use the predefined function `run_controller_in_task(controller : Controller, task : Task, n_episodes : int = 1) -> Feedback` to play the controller in the task.
It is possible to use the controller in your code without using this function for debugging or anything you found useful to do, but it is not recommended as we don't see the interest of it.

- **Knowledge**: a class that contains information about the environment, the tasks, or the agentic framework itself. It is used to maintain any information that you judge relevant and \
worth of being stored in the knowledge base. You interact with them only through the code-edit action, since they are not executable objects.
At the beginning of each conversation, these knowledges will be displayed for you in a "Knowledge" section, so you can use them to help you solve the tasks.
{self.code_tag(self.code_item_knowledge)}
To create/modify a knowledge, instanciate a well-named instance of the class `Knowledge` in your code-edit action. E.g. `knowledge_observation_shape = Knowledge(content="The shape of the observation is (64, 64, 3).")`
To delete a knowledge, write ```del item_name``` in your code-edit action.


## Step-by-step process :
At each call, we ask you to answer first by eventually reason about what you should do, and then submit an action under the form of a code block.
When you submit an action, it's result will be instantly available to you (e.g. success of a coding action, or the result of a test/information retrieval action).
Then you can submit a new action.


## Conversation refreshing mechanism :
This conversation (system prompt and succession of actions/answers) is limited by the context window of the LLM and slow inference time with increasing length.
For this reason, you will have to periodically refresh the conversation by using the refresh action through the execution of the following code : ```agentic.refresh_conversation()``.
This will reset the conversation by removing any traces of actions/answers, leaving only the system prompt and the code base.
Consequently, it is VERY IMPORTANT to take notes of any relevant information you obtained during the conversation, either in the knowledge or in the docstrings of code items, before refreshing the conversation.
After each answer that you consider important to keep trace of, you should store or update the notes of the involved items, otherwise this information will be lost for future inferences.


## Code Actions :
You can perform actions through code blocks between corresponding tags. Actions are divided in two categories:

1) **Editing actions**: these actions allow you to edit the code base / knowledge base by creating/modifying/deleting items.
This will happend when you (re)define an item, or if you delete one with ```del item_name```.
You can use them through the `code:edit` tag, e.g.:
<code:edit>
```python
class NewClass(...):  # to create a new class
    pass

def new_function(...):  # to create a new function
    pass

new_variable = ...  # to create a new variable

class OldClass(...):  # to modify an existing class (works the same with function or variable)
    pass

del old_item_name  # to delete an item (class, function, variable)
```
</code:edit>


Feedback : the feedback of the code-edit action will be available in the next answer, and will tell you if the editing action was successful or not.
If the editing action was successful, the item will be available in the code base for the next actions.
If the editing action was not successful, the whole editing action will be cancelled, and the error will be displayed in the next answer.

Notes:
    - You can create/modify/delete as many items as you want in a single code-edit action, but we advise you to go step by step and not to do too much at once, \
as it can be hard to debug if something goes wrong.
    - If your code requires some dependencies regarding other items, you don't need to import them, they will be imported automatically.
    - You must take care of not creating any circular dependency between items, as it will lead to an error.
    - If you need to import some basic python libraries (e.g. `import numpy as np`), you must do it at the beginning of the code-edit action.


2) **Execution actions**: these actions allow you to execute the code base / knowledge base by running items.
This will happen by calling an item or if you use certain predefined functions in a code-exec action.
You can use them through the `code:exec` tag, e.g.:
<code:exec>
```python
# example : instantiate a controller, run it in a task, and print the feedback back to you
controller = MyController(...)
task = agentic.get_task("task_name")  # get the task from the environment
feedback = run_controller_in_task(controller, task, n_episodes=10) 
print(feedback)  # to print the feedback of the controller

# example : test that a function is working correctly
result = my_function(...)
assert result == expected_result, "The function did not return the expected result."
```
</code:exec>

Output : the stdout/stderr of the code-exec action will be available in the next answer. It will contain the output of the code executed, as well as any error that may have occurred during the execution.
You can't use any variable defined in the code-exec action in other actions, the variables are not saved anywhere.

## Special code-exec actions:
You will have access to some special code-exec actions that you will have to use at some point :

- Refresh conversation : running ```agentic.refresh_conversation()``` in a code-exec action will reset the conversation, removing all traces of actions/answers, leaving only the system prompt and the code base. \
You must use this action periodically to avoid the context window of the LLM to be full, which would lead to a failure of the inference, but you must be sure all relevant information is stored in the notes of the items before doing so, as it will be lost otherwise.

- Editing docstring only : if you wish to edit the docstring of an item without modifying its code, you can use the .edit_docstring() method, e.g.:
<code:exec>
```python
agentic.edit_docstring("item_name", "This is the new docstring of the item.")
```
</code:exec>
This is usefull to quickly update any information you want to keep trace of in the item, without modifying its code.

- Getting a task : you can get a task from the environment by using the predefined function ```agentic.get_task(task_name: str) -> Task```.

- Playing a controller in a task : you can play a controller in a task using the predefined function ```run_controller_in_task(controller : Controller, task : Task, n_episodes : int = 1) -> Feedback```.
<code:exec>
```python
task = agentic.get_task("task_name")  # get the task from the environment
controller = MyController(...)
feedback = agentic.run_controller_in_task(controller, task=task, n_episodes=10)
print(feedback)
```
</code:exec>
Playing a controller in a task will give you feedback on the controller and help you design better controllers in the future and accumulate knowledge about the environment and the tasks.

- Submitting a controller to a task : once you have identified a satisfying controller for a task, you can submit it to the task by running ```agentic.submit_controller_to_task(controller, task)```.
This will save it as a solution to the task and provide you a reward based on the quality of the solution relative to your previous solution.
"""
            self.dict_system_prompts["instructions"] = prompt_instructions

        # Env prompt
        if "env" in self.list_prompt_keys:
            description_env = env.get_textual_description()
            prompt_env = (
                "## General description of the environment\n" f"{description_env}"
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

        # === Initialize the codebase manager ===
        self.codebase_manager = CodebaseManager()
        code_initial_codebase = open("agent/agentic/initial_codebase.py").read()
        self.codebase_manager.edit_code(
            code_initial_codebase, allow_several_top_defs=True
        )

    # ============ Interface methods =================

    def is_done(self):
        return self.timestep >= self.n_steps_max

    def step(self):
        with RuntimeMeter("step"):

            if self.is_conversation_fresh:
                self.is_conversation_fresh = False
                self.idx_conversation += 1
                self.timestep_in_conv = 0

                # Update the tasks completion status
                tasks_from_env: List[Task] = self.env.get_current_tasks()
                for task in tasks_from_env:
                    if task.get_name() not in self.tasks_status:
                        # Initialize the task status
                        self.tasks_status[task.get_name()] = TaskStatus(
                            name=task.get_name(),
                            task=task,
                        )
                # Build the prompt system
                prompt_system = self.build_system_prompt(
                    dict_prompts={
                        **self.dict_system_prompts,
                        "tasks": self.get_tasks_prompt(),
                        "status": self.get_status_prompt(),
                        "codebase": self.get_codebase_prompt(),
                    }
                )
                self.messages = [{"role": "system", "content": prompt_system}]
                self.log_prompt(prompt_system)

            # Generate the answer from the LLM
            answer = self.llm.generate(messages=self.messages, n=1)[0]
            self.messages.append({"role": "assistant", "content": answer})
            self.log_answer(answer)

            # Apply the answer
            with StreamCapture("[Output]") as stream_capture:
                try:
                    # Extract the code snippet from the answer
                    code, mode = self.extract_code_snippet(answer)
                    self.log_code(code)

                    # Edit mode
                    if mode == "EDIT":
                        print("Editing codebase/knowledge base...")
                        self.codebase_manager.edit_code(code)

                    # Exec mode
                    elif mode == "EXEC":
                        print("Executing code...")
                        self.codebase_manager.execute_code(
                            code,
                            variables={
                                "agentic": self,
                                **self.env_variables,
                            },
                        )

                    else:
                        raise ValueError(
                            f"Invalid mode '{mode}' extracted from the answer. Expected 'EDIT' or 'EXEC'."
                        )

                except CodeExtractionError as e:
                    print(f"Error extracting code snippet: {str(e)}")
                    
                except CodeExecutionError as e:
                    print(f"Error during code editing/executing: {str(e)}")
                
            # Add the output to the messages
            output = f"<output>\n{stream_capture.get_output()}</output>"
            self.messages.append({"role": "system", "content": output})
            self.log_output(output, mode=mode)

        # Log runtime metrics
        metrics_runtime = self.get_runtime_metrics()
        self.logger.log_scalars(metrics_runtime, step=self.timestep)

        # Move forward the iter counter
        self.timestep += 1
        self.timestep_in_conv += 1
        breakpoint()

    # ============ Helper methods =================

    def build_system_prompt(
        self,
        dict_prompts: Dict[str, str],
    ) -> str:
        """Build the prompt to send to the LLM.

        Args:
            dict_prompts (Dict[str, str]): a dictionary containing the different prompts.

        Returns:
            str: the prompt to send to the LLM
        """
        list_prompt = []
        for key_prompt in self.list_prompt_keys:
            if key_prompt not in dict_prompts:
                raise ValueError(
                    f"Key '{key_prompt}' not found in the provided dictionary."
                )
            else:
                list_prompt.append(dict_prompts[key_prompt])

        prompt = "\n\n".join(list_prompt)
        return prompt

    def extract_code_snippet(self, text: str):
        """Extracts a single Python code snippet wrapped in <code:exec> or <code:edit> with ```python ... ``` inside.

        Args:
            text (str): The text containing the code snippet.

        Returns:
            (code: str, mode: str) if valid (mode is "EXEC" or "EDIT")
            Raises ValueError if conditions are not met.
        """

        # Match <code:exec> or <code:edit> block
        matches = re.findall(
            r"<code:(exec|edit)>\s*```python\n(.*?)```[\s\n]*</code:\1>",
            text,
            re.DOTALL,
        )

        if len(matches) == 0:
            raise CodeExtractionError(
                "No valid <code:exec> or <code:edit> block found with ```python."
            )
        elif len(matches) > 1:
            raise CodeExtractionError(
                "Multiple code blocks found — only one is allowed."
            )

        mode, code = matches[0]
        return code.strip(), mode.upper()

    def code_tag(self, code: str) -> str:
        """Add the python code tag to some code."""
        return f"```python\n{code}\n```"

    def get_tasks_prompt(self) -> str:
        """Get a summary of the tasks available in the environment."""
        result = "## Tasks available in the environment:\n\n"
        if len(self.tasks_status) == 0:
            raise ValueError(
                "No tasks available in the environment. Please check the environment configuration."
            )
        result += "\n\n".join(
            [str(task_status) for task_status in self.tasks_status.values()]
        )
        return result

    def get_status_prompt(self) -> str:
        """Get a summary of the current status of the agent."""
        delta_time = time.time() - self.time_start
        time_str = time.strftime("%H:%M:%S", time.gmtime(delta_time))
        if self.idx_conversation >= 1:
            info_refresh = f"Output : Conversation has been refreshed. This is the first message of the new conversation (#{self.idx_conversation})."
        else:
            info_refresh = f"Output : Agentic started."
        return (
            "## Current status of the agent:\n"
            f"Runtime : you are running for {time_str}.\n"
            f"Conversation index : This is the conversation number {self.idx_conversation}.\n"
            f"Current timestep : The total number of elementary steps taken since the beginning of the run is {self.timestep}.\n"
            f"{info_refresh}"
        )

    def get_codebase_prompt(self) -> str:
        """Get a summary of the current code base."""
        return "## Current code base:\n" f"{str(self.codebase_manager)}"

    def get_performance_metric_from_feedback(self, metrics: Dict[str, float]) -> Tuple[float, bool]:
        THRESHOLD_SUCCESS = 0.9
        performance_metric = metrics["success/rate"] + metrics["reward/mean"]
        performance_metric = float(performance_metric) 
        is_done = (metrics["success/rate"] >= THRESHOLD_SUCCESS)
        return performance_metric, is_done

    # ============ Action methods =================

    def refresh_conversation(self):
        """Refresh the conversation by resetting the messages and the conversation index."""
        self.is_conversation_fresh = True
        print(
            f"Refreshing conversation, starting conversation {self.idx_conversation+1} at timestep {self.timestep}."
        )

    def edit_docstring(self, item_name: str, new_docstring: str):
        """Edit the docstring of an item in the code base."""
        pass  # TODO : implement this method to edit the docstring of an item in the code base.

    def get_task(self, task_name: str) -> Task:
        """Get a task from the environment by its name."""
        if task_name not in self.tasks_status:
            raise ValueError(
                f"Task '{task_name}' not found in the tasks status. Available tasks: {list(self.tasks_status.keys())}"
            )
        return self.tasks_status[task_name].task

    def run_controller_in_task(
        self, controller: Controller, task: Task, n_episodes: int = 1
    ) -> FeedbackAggregated:
        """Run a controller in a task for a specified number of episodes."""
        feedback_agg_over_episodes = play_controller_in_task(
            controller,
            task,
            n_episodes=n_episodes,
            is_eval=False,
            log_subdir=f"conversation_{self.idx_conversation}/run_output/_step_{self.timestep_in_conv}_task_{sanitize_name(task.get_name())}",
        )
        # Log the feedback
        task_name_sanitized = sanitize_name(task.get_name())
        metrics = feedback_agg_over_episodes.get_metrics(prefix=f"run_controller/task_{task_name_sanitized}")  # success/rate, submmission/task_X/success/rate
        performance_metric, is_done = self.get_performance_metric_from_feedback(
            metrics
        )
        metrics[f"submissions/task_{task_name_sanitized}/performance_metric"] = performance_metric
        metrics[f"submissions/task_{task_name_sanitized}/is_done"] = is_done
        with StreamCapture("[Log scalars]", log_to_cli=False) as _:
            self.logger.log_scalars(
                metrics, step=self.timestep
            )
        self.log_metrics(metrics, mode = "RUN")
        
        # Return the aggregated feedback
        return feedback_agg_over_episodes
    
    def submit_controller_to_task(
        self, controller: Controller, task: Task
    ) -> FeedbackAggregated:
        """Submit a controller to a task and get the feedback."""
        task_name = task.get_name()
        task_name_sanitized = sanitize_name(task_name)
        N_EPISODES_SUBMIT_CONTROLLER = 20
        self.tasks_status[task_name].n_submissions += 1
        print(f"Controller submitted to task '{task_name}'.")
        # Play the controller in the task and get the feedback
        feedback_agg_over_episodes = play_controller_in_task(
            controller,
            task,
            n_episodes=N_EPISODES_SUBMIT_CONTROLLER,
            is_eval=False,
            log_subdir=f"conversation_{self.idx_conversation}/submission_output/step_{self.timestep_in_conv}_task_{task_name_sanitized}",
        )
        # Compute the performance metric from the feedback
        metrics = feedback_agg_over_episodes.get_metrics(prefix=f"submissions/task_{task_name_sanitized}")  # success/rate, submmission/task_X/success/rate
        performance_metric, is_done = self.get_performance_metric_from_feedback(
            metrics
        )
        print(
            f"Obtained performance metric: {performance_metric}. Task done: {is_done}."
        )

        # Check if the performance metric improved
        previous_performance_metric = self.tasks_status[task_name].performance_metric
        if previous_performance_metric is None:
            previous_performance_metric = 0.0
        previous_is_done = self.tasks_status[task_name].is_done

        if performance_metric > previous_performance_metric:
            # Update the task status
            self.tasks_status[task_name].performance_metric = performance_metric
            # Provide reward if performance metric improved
            print(
                f"REWARD OBTAINED: {performance_metric - previous_performance_metric:.2f} (performance metric improvement!)"
            )
        elif performance_metric == previous_performance_metric:
            print(f"No performance metric improvement.")
        else:
            # Performance metric decreased, do not update the task status
            print(
                f"WARNING: Performance metric decreased from {previous_performance_metric} to {performance_metric}. Task status not updated."
            )

        # Check if the task is done
        if is_done and not previous_is_done:
            print(f"REWARD OBTAINED: 1.0 (task newly completed!)")
            self.tasks_status[task_name].is_done = True
        elif is_done and previous_is_done:
            print(
                f"Task '{task_name}' is completed, but no reward obtained (already completed in the previous version)."
            )
        elif not is_done and previous_is_done:
            print(
                f"WARNING: Task '{task_name}' is not completed anymore, but it was in the previous version."
            )
        else:
            print(f"Task '{task_name}' is still not completed yet, no reward obtained.")

        # Also print the feedback
        print(
            f"Feedback aggregated over {N_EPISODES_SUBMIT_CONTROLLER} controllers:\n{feedback_agg_over_episodes}"
        )
        
        # Log the metrics
        metrics[f"submissions/task_{task_name_sanitized}/performance_metric"] = performance_metric
        metrics[f"submissions/task_{task_name_sanitized}/is_done"] = is_done
        with StreamCapture("[Log scalars]", log_to_cli=False) as _:
            self.logger.log_scalars(
                metrics, step=self.timestep
            )
        self.log_metrics(metrics, mode = "SUBMIT")
        
    # ============ Log methods =================

    def log_conversation(self):
        prompt_templated_list = [
            f"{msg['role']}:\n{msg['content']}" for msg in self.messages
        ]
        prompt_templated = "\n\n\n".join(prompt_templated_list)
        self.log_as_texts(
            {
                f"conv.txt": prompt_templated,
            },
            log_subdir=f"conversation_{self.idx_conversation}",
            verbose=False,
        )

    def log_prompt(self, prompt: str):
        self.log_as_texts(
            {
                f"prompt.txt": prompt,
            },
            log_subdir=f"conversation_{self.idx_conversation}",
        )
        self.log_conversation()

    def log_answer(self, answer: str):
        self.log_as_texts(
            {
                f"step_{self.timestep_in_conv}_answer.txt": answer,
            },
            log_subdir=f"conversation_{self.idx_conversation}",
            verbose=False,
        )
        self.log_conversation()

    def log_code(self, code: str):
        # dont log conv here, it is already logged in log_answer
        self.log_as_texts(
            {
                f"step_{self.timestep_in_conv}_code.py": code,
            },
            log_subdir=f"conversation_{self.idx_conversation}",
            verbose=False,
        )

    def log_output(self, feedback_info: str, mode: str = None):
        if mode is None:
            name = f"step_{self.timestep_in_conv}_output.txt"
        else:
            name = f"step_{self.timestep_in_conv}_output_{mode.lower()}.txt"
        self.log_as_texts(
            {
                name: feedback_info,
            },
            log_subdir=f"conversation_{self.idx_conversation}",
            verbose=False,
        )
        self.log_conversation()

    def log_metrics(self, metrics: Dict[str, float], mode : str = None):
        """Log the metrics in the logger."""
        if mode is None:
            name = f"step_{self.timestep_in_conv}_metrics.txt"
        else:
            name = f"step_{self.timestep_in_conv}_metrics_{mode.lower()}.txt"
        self.log_as_texts(
            {
                name: metrics,
            },
            log_subdir=f"conversation_{self.idx_conversation}",
            verbose=False,
        )