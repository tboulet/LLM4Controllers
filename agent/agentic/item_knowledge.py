class Knowledge:
    """
    This class represents a knowledge that can store informations about anything usefull to solve the environment in the agentic framework.
    Note it can also be used to store information about the mechanism of the agentic framework itself, if it is judged useful.
    """
    def __init__(self, content : str):
        """
        Initialize the knowledge with the given content.
        
        Args:
            content (str): The initial content of the knowledge.
        """
        self.content = content