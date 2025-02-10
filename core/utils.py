import traceback

def get_error_info(e):
    stack_trace = traceback.format_exc()  # Capture the full stack trace as a string
    relevant_stack_trace = "\n".join(line for line in stack_trace.splitlines() if "<string>" in line)
    error_info = f"Stack Trace:\n{relevant_stack_trace} \nError Output:\n{e}"
    return error_info

dict_warning_messages = set()
def one_time_warning(warning_message):
    if warning_message not in dict_warning_messages:
        print(f"WARNING : {warning_message}")
        dict_warning_messages.add(warning_message)
        