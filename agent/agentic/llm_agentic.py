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
import io
import sys
import traceback


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
        self.idx_conversation: int = 0
        self.is_conversation_fresh: bool = True
        self.env_variables: Dict[str, Any] = {}
        self.base_scope = {}
        exec(open("agent/llm_hcg/base_scope.py").read(), self.base_scope)

        # Initialize LLM
        print("Initializing LLM...")
        name_llm = config["llm"]["name"]
        config_llm = config["llm"]["config"]
        self.llm = llm_name_to_LLMClass[name_llm](**config_llm, logger=logger)

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
            prompt_instructions = f"""You are Agentic, an AI agent that must solve an task-based environment.
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
In particular, you can use the predefined function `play_controller_in_task(controller : Controller, task : Task, n_episodes : int = 1) -> Feedback` to play the controller in the task.
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
For this reason, you will have to periodically refresh the conversation by using the `refresh` action.
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
ctrl = MyController(...)  
feedback = play_controller_in_task(ctrl, task=task, n_episodes=10) 
print(feedback)  # to print the feedback of the controller

# example : test that a function is working correctly
result = my_function(...)
assert result == expected_result, "The function did not return the expected result."
```
</code:exec>

Feedback : the feedback of the code-exec action will be available in the next answer. It will contain the stdout of the code executed, as well as any error that may have occurred during the execution.
You can't use any variable defined in the code-exec action in other actions, the variables are not saved anywhere.

Special code-exec features :

- Refresh conversation : running ```agentic.refresh()``` in a code-exec action will reset the conversation, removing all traces of actions/answers, leaving only the system prompt and the code base. \
You must use this action periodically to avoid the context window of the LLM to be full, which would lead to a failure of the inference, but you must be sure all relevant information is stored in the notes of the items before doing so, as it will be lost otherwise.

- Editing docstring only : if you wish to edit the docstring of an item without modifying its code, you can use the .edit_docstring() method of the item, e.g.:
<code:exec>
```python
my_item.edit_docstring("This is the new docstring of the item.")
```
</code:exec>
This is usefull to quickly update any information you want to keep trace of in the item, without modifying its code.

- Environment code : the environment prompt will explain you how to create Task objects. \
You can then play a controller in a task using the predefined function `play_controller_in_task(controller : Controller, task : Task, n_episodes : int = 1) -> Feedback`. Example:
<code:exec>
```python
task = EnvClassExplainedByEnvPrompt(...)
```
</code:exec>
Playing a controller in a task will serve two purposes : 1) it is an attempt to solve the task, which is your final objective (solving all tasks), and 2) it will give you feedback on the controller and help you design better controllers in the future and accumulate knowledge about the environment and the tasks.
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

    def build_prompt(
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

    def step(self):
        with RuntimeMeter("step"):

            if self.is_conversation_fresh:
                self.is_conversation_fresh = False
                # Extract the env usage explanation and the env variables to set
                prompt_env_usage_explanation, self.env_variables = (
                    self.env.get_env_usage_explanation_and_variables()
                )
                prompt_env_usage_explanation = (
                    "## Environment usage explanation:\n"
                    f"{prompt_env_usage_explanation}"
                )
                # Build the prompt
                prompt_system = self.build_prompt(
                    dict_prompts={
                        **self.dict_system_prompts,
                        "env_usage": prompt_env_usage_explanation,
                        # TODO : other prompts related to code base here
                    }
                )
                self.messages = [{"role": "system", "content": prompt_system}]
            else:
                raise NotImplementedError(
                    "Conversation refreshing mechanism is not implemented yet."
                )


            # Turn messages into a readable conversation and log it
            prompt_templated_list = [
                f"{msg['role']}:\n{msg['content']}" for msg in self.messages
            ]
            prompt_templated = "\n\n\n".join(prompt_templated_list)
            self.log_as_texts(
                {
                    f"conv.txt": prompt_templated,
                },
                log_subdir=f"conversation_{self.idx_conversation}",
            )

            # Generate the answer from the LLM
            answer = self.llm.generate(messages=self.messages, n=1)[0]
            self.log_as_texts(
                {
                    f"step_{self.timestep}_answer.txt": answer,
                },
                log_subdir=f"conversation_{self.idx_conversation}",
            )
            
            # Apply the answer
            try:
                # Extract the code snippet from the answer
                code, mode = self.extract_code_snippet(answer)
                self.log_as_texts(
                    {
                        f"step_{self.timestep}_code.py": code,
                    },
                    log_subdir=f"conversation_{self.idx_conversation}",
                )
                
                # Edit mode
                if mode == "EDIT":
                    raise
                
                # Exec mode
                elif mode == "EXEC":
                    output = self.execute_exec_code(
                        code, self.env_variables
                    )
                    self.log_as_texts(
                        {
                            f"step_{self.timestep}_output.txt": output,
                        },
                        log_subdir=f"conversation_{self.idx_conversation}",
                    )
                    feedback_info = f"<feedback>\n{output}\n</feedback>"
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": feedback_info,
                        }
                    )
                    
                else:
                    raise
                    
            except CodeExtractionError as e:
                raise ControllerExecutionError(
                    f"Error while extracting code snippet from answer: {e}"
                )
                
                
        # Log runtime metrics
        metrics_runtime = self.get_runtime_metrics()
        self.logger.log_scalars(metrics_runtime, step=self.timestep)

        # Move forward the iter counter
        self.timestep += 1


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
            raise CodeExtractionError("Multiple code blocks found — only one is allowed.")

        mode, code = matches[0]
        return code.strip(), mode.upper()



    def execute_exec_code(self, code: str, env_variables: dict) -> str:
        """
        Executes a code string with access to env_variables.
        Captures interleaved stdout and stderr output into a single stream.

        Returns:
            str: Captured combined output (stdout + stderr in order).
        """
        buffer = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buffer  # Redirect both to same stream

        try:
            exec(code, env_variables)
        except Exception:
            traceback.print_exc()

        sys.stdout, sys.stderr = old_stdout, old_stderr
        return buffer.getvalue()



    def is_done(self):
        return self.timestep >= self.n_steps_max

    def code_tag(self, code: str) -> str:
        """Add the python code tag to some code."""
        return f"```python\n{code}\n```"
