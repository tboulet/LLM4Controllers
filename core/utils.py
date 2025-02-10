import traceback

def get_error_info(e):
    stack_trace = traceback.format_exc()  # Capture the full stack trace as a string
    relevant_stack_trace = "\n".join(line for line in stack_trace.splitlines() if "<string>" in line)
    error_info = f"Stack Trace:\n{relevant_stack_trace} \nError Output:\n{e}"
    return error_info
