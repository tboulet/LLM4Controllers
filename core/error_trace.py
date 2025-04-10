
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