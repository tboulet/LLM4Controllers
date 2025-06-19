from transformers.utils import get_json_schema

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"  # or "HuggingFaceH4/Qwen3-0.6B" for the hosted version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)





def get_current_temperature(location: str):
    """
    Gets the temperature at a given location.

    Args:
        location: The location to get the temperature for
    """
    return 22.0 

tools = [get_current_temperature]    

chat = [
    {"role": "user", "content": "Hey, what's the weather like in Paris right now?"}
]
print(tokenizer.chat_template)
tool_prompt = tokenizer.apply_chat_template(
    chat, 
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=False
)

print(tool_prompt)