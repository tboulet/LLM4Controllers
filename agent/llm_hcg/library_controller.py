import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from agent.base_agent import BaseAgent, Controller
from agent.llm_hcg.graph_viz import ControllerVisualizer
from core.task import TaskDescription
from core.utils import get_error_info
from env.base_meta_env import BaseMetaEnv, Observation, ActionType, InfoDict
from abc import ABC, abstractmethod
import enum
import random
from typing import Any, Dict, Tuple, Union


class ControllerData:
    """Represents the data of a controller in the controller library."""

    def __init__(
        self,
        code: str,
        controller: Controller = None,
    ):
        """Initialize the ControllerData object.

        Args:
            code (str): the code of the controller
            controller (Controller): the controller object (optional for now)
        """
        self.code = code
        self.controller = None

    def __repr__(self):
        return self.code  # TODO : edit this to only show docstrings/resumes


class ControllerLibrary:
    """
    This class represents the controller library of the agent.
    It implements method to update it and to query it.
    """

    def __init__(self, config_agent: Dict, visualizer: ControllerVisualizer):
        # Initialize controller library
        print("Initializing controller library...")
        self.config = config_agent["config_controllers"]
        self.controllers: Dict[str, ControllerData] = {}  # map controller_name to code
        # Initialize visualizer
        self.visualizer = visualizer
        # Initialize the controller library with the controllers defined in the initial_controllers folder
        which_initial_controllers: str = self.config["which_initial_controllers"]
        path_to_initial_controllers: str = "agent/llm_hcg/initial_PCs"
        if which_initial_controllers == "none":
            pass
        elif which_initial_controllers == "specific":
            for init_controller_file_name in self.config["initial_controllers"]:
                f = open(
                    os.path.join(
                        path_to_initial_controllers, init_controller_file_name
                    ),
                    "r",
                )
                code = f.read()
                self.add_primitive_controller(code)
                f.close()

        elif which_initial_controllers == "all":
            for file_name in os.listdir(path_to_initial_controllers):
                if file_name.endswith(".py"):
                    f = open(os.path.join(path_to_initial_controllers, file_name), "r")
                    code = f.read()
                    self.add_primitive_controller(code)
                    f.close()
        else:
            raise ValueError(
                f"Unknown value for which_initial_controllers: {which_initial_controllers}"
            )

    def add_primitive_controller(self, code: str):
        """Add a controller to the controller library from a code string.
        The code should contain imports and a class definition.
        The name of the class and the class definition will be extracted from the file as strings.
        The name of the class will be used as the key in the controller library, and the class definition as the value.

        Args:
            code (str): the code of the controller

        Raises:
            ValueError: if the controller is already present in the controller library (same name)
        """
        class_name = self.extract_class_name(code)
        if class_name in self.controllers:
            raise ValueError(
                f"Controller {class_name} was attempted to be added to the controller library but is already present."
            )
        if "from controller_library" in code:
            raise ValueError(
                "You are not allowed to import controllers from the controller_library module when creating a new primitive controller, only when creating a specialized controller for inference/refactoring."
            )
        # Create a DataController object to store information about the controller
        data = ControllerData(
            code=code,
        )
        # Add the controller's data to the controller library
        self.controllers[class_name] = data
        # Add the controller to the visualizer
        self.visualizer.add_PCs({class_name: code})

    def __repr__(self):
        if len(self.controllers) == 0:
            res = "The controller library is empty for now.\n"
        else:
            res = "The controller library contains the following controllers:\n\n"
            for name_controller, data_controller in self.controllers.items():
                res += f"{name_controller}:\n```python\n{data_controller.code}\n```\n\n"
        return res

    def extract_class_name(self, code: str) -> str:
        """Extract the name of the class from the code of the controller.

        Args:
            code (str): the code of the controller

        Raises:
            ValueError: if no class definition is found in the code

        Returns:
            str: the name of the class
        """
        match = re.search(r"class\s+(\w+)\s*\(", code)
        if match:
            class_name = match.group(1)  # Extracts the class name
            return class_name
        raise ValueError("No class definition found in the file content.")
