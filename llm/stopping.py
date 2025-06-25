from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch

# Load model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the stop sequence
stop_sequence = "</code>"


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt):
        self.target_sequence = target_sequence
        self.prompt = prompt

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt, "")
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation

        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self



prompt = "Here are 3 examples of codes : <code>print('Hello, world!')</code><code>print('This is a test')</code><"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Apply the stopping criteria
stopping_criteria = MyStoppingCriteria(target_sequence=stop_sequence, prompt=prompt)

# Generate
output = model.generate(
    **inputs,
    max_new_tokens=100,
    stopping_criteria=stopping_criteria,
    do_sample=True,
    temperature=0.7
)

# Decode and print
print(tokenizer.decode(output[0], skip_special_tokens=False))

breakpoint()
