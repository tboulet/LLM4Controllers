import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from core.task import TaskRepresentation
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union


class KnowledgeBase:
    """
    This class represents the knowledge base of the LLM-HCG agent. It
    implements method to update it and to query it.

    It is based of different components:
    - the controller library (for now only thing implemented)
    - the logical functions library
    - the procedural knowledge of the agent
    - the hypothesis of the agent
    """

    def __init__(self, config_agent: Dict, namespace : Dict[str, Any]):
        # Initialize controller library
        print("Initializing KnowledgeBase...")
        self.namespace = namespace
        self.config_controllers = config_agent["config_controllers"]
        self.controller_library: Dict[str, str] = (
            {}
        )  # map controller_name to field to value
        if self.config_controllers["do_use"]:
            print("\tInitializing controller library...")
            which_initial_controllers: str = self.config_controllers[
                "which_initial_controllers"
            ]
            path_to_initial_controllers: str = os.path.join(
                "agent", "llm_hcg", "initial_controllers"
            )
            if which_initial_controllers == "none":
                pass
            elif which_initial_controllers == "specific":
                for init_controller_file_name in self.config_controllers[
                    "initial_controllers"
                ]:
                    self.add_controller_from_file(
                        os.path.join(
                            path_to_initial_controllers, init_controller_file_name
                        )
                    )

            elif which_initial_controllers == "all":
                for file_name in os.listdir(path_to_initial_controllers):
                    if file_name.endswith(".py"):
                        self.add_controller_from_file(
                            os.path.join(
                                path_to_initial_controllers,
                                file_name,
                            )
                        )
            else:
                raise ValueError(
                    f"Unknown value for which_initial_controllers: {which_initial_controllers}"
                )

    def add_controller_from_file(self, file_name: str):
        """Add a controller to the controller library from a file.
        The file should contain imports (who will be ignored) and a class definition.
        The name of the class and the class definition will be extracted from the file as strings.
        The name of the class will be used as the key in the controller library, and the class definition as the value.

        Args:
            file_name (str): the name of the file containing the controller definition

        Raises:
            ValueError: if the controller is already present in the controller library (same name)
        """
        with open(file_name, "r") as f:
            file_content = f.read()
            class_name, class_def = self.extract_class_info(file_content)
            if class_name in self.controller_library:
                raise ValueError(
                    f"Controller {class_name} was attempted to be added to the controller library but is already present."
                )
            # Add the controller's code to the controller library
            self.controller_library[class_name] = class_def
            # Run the controller's code in the namespace to make it available
            exec(class_def, self.namespace)
            
    def __repr__(self):
        res = "KnowledgeBase:\n"
        # Controllers
        if self.config_controllers["do_use"]:
            if len(self.controller_library) == 0:
                res += "\t- Controllers : the controller library is empty.\n"
            else:
                res += (
                    "\t- Controllers : these controllers will be imported automatically and you can use them in your code as sub-controllers. "
                    "For example you can instantiate them in your __init__ or act method and then use them in your act method. "
                    "Please take care of the signatures of the methods of the controllers you use. "
                    "\n\n"
                )
                for controller_name, controller_def in self.controller_library.items():
                    res += f"{controller_name}:\n{controller_def}\n\n"
        return res
    
    def extract_class_info(self, file_content: str):
        match = re.search(r"\bclass\s+(\w+)\s*\(.*?\):([\s\S]+)", file_content)
        if match:
            class_name = match.group(1)  # Extracts the class name
            class_definition = (
                "class " + match.group(0).split("class", 1)[1]
            )  # Extracts the full class definition
            return class_name, class_definition
        raise ValueError("No class definition found in the file content.")
