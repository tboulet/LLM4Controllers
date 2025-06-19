from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Any, List, Optional
import numpy as np


class Observation:
    pass


class ActionType:
    pass


class InfoDict(Dict[str, Any]):
    pass



class ErrorTrace:
    def __init__(self, error_message : str):
        """
        Initialize the ErrorTrace object.
        
        Args:
            error_message (str): the message of the error
        """
        self.error_message = error_message
    
    def __repr__(self):
        return self.error_message
    
    
class TextualInformation:
    def __init__(self, text: str):
        """
        Initialize the TextualInformation object.
        
        Args:
            text (str): the text of the information
        """
        self.text = text
    
    def __repr__(self):
        return self.text
    
    
class CodeExtractionError(Exception):
    pass


class ControllerExecutionError(Exception):
    pass