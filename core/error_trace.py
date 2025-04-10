
class ErrorTrace:
    def __init__(self, error_type : str, error_message : str):
        """
        Initialize the ErrorTrace object.
        
        Args:
            error_type (str): the type of the error
            error_message (str): the message of the error
        """
        self.error_type = error_type
        self.error_message = error_message
    
    def __repr__(self) -> str:
        return f"Error type : {self.error_type}.\nError message : {self.error_message}"