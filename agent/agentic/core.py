from agent.base_controller import Controller
from core.play import play_controller_in_task

REFRESH_CONVERSATION = "refresh_conversation"

def run_controller_in_task(controller: Controller, task: str, n_episodes: int = 1):
    """
    Run a controller in a task for a specified number of episodes.
    
    Args:
        controller (Controller): The controller to run.
        task (str): The task to run the controller in.
        n_episodes (int): The number of episodes to run.
    """
    play_controller_in_task(controller, task, n_episodes, is_eval=False, log_subdir="dump_run_controller")


def refresh_conversation():
    """
    Refresh the conversation state.
    """
    return REFRESH_CONVERSATION