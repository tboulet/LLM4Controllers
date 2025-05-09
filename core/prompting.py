from collections import defaultdict
import os
import re
from typing import Dict, List

class Prompt:
    def __init__(self, *prompts : str):
        """
        Initialize the Prompt object.

        Args:
            *prompts (List[str]): the prompts to be used
        """
        self.content = "\n"