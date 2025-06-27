from agent.agentic.items import Controller, Knowledge
from agent.agentic.types import Observation, ActionType

import numpy as np

knowledge_observation_shape = Knowledge("""
The observation in the environment is a dictionnary with the following keys:
- 'image' : a numpy array of shape (height, width, 3) representing the image.
- 'direction' (int) : the direction the agent is facing, 0/1/2/3 for up/right/down/left.
- 'mission' (str) : the mission string describing the task to be accomplished in the episode.
    It follows a certain template accessible in the task description, such as 'go towards direction {direction}'
""")

knowledge_strategy_1 = Knowledge("""
I should try to solve this task-based environment challenge in a structured and step-by-step manner.
This is a problem where I will interact with code-action/answer pairs, so I should think about the long-term success and store relevant and trustworthy information in my knowledge base.
I should try to understand as much as possible how is the environment working, and how to interact with it.
""")



knowledge_actions = Knowledge("""
The actions I can take in the environment are:
- 'forward' : move forward in the direction I am facing.
- 'left' : turn left.
- 'right' : turn right.
- 'backward' : turn around.
- 'pickup' : pick up the object I am facing (if any).
- 'drop' : drop the object from my inventory (if any) in front of me.
- 'toggle' : toggle/activate an object in front of me.
- 'done' : end the episode if the task is completed.
""")


class MoveForwardController(Controller):

    def act(self, observation: Observation) -> ActionType:
        return "forward"
    

COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

def find_object_position(observation: Observation, object_type: str) -> ActionType:
    """
    Find the position of a specific object type in the observation.

    Args:
        observation (Observation): The observation containing the environment state.
        object_type (str): The type of object to find (e.g., 'goal').

    Returns:
        ActionType: The action to take, which is to find the position of the specified object.
    """
    image_object = observation['image'][..., 0]  # Assuming the first channel contains object types
    position = np.argwhere(image_object == OBJECT_TO_IDX[object_type])[0]
    return position if position.size > 0 else None


class FindFGoalPosition(Controller):
    """
    Controller to find the position of the goal in the observation.
    """

    def act(self, observation: Observation) -> ActionType:
        """
        Act to find the goal position.

        Args:
            observation (Observation): The observation containing the environment state.

        Returns:
            ActionType: The action to take, which is to find the goal position.
        """
        # Assuming 'goal' is represented by a specific object type in the observation
        return find_object_position(observation, 'goal')
    