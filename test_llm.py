import os
from transformers import AutoTokenizer

from llm.llm_from_hf import LLM_from_HuggingFace

# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
# assert tokenizer.chat_template is not None

# # OK to use
# prompt = tokenizer.apply_chat_template(messages, tokenize=False)
# print(prompt)

# model_name = "/lustre/fsn1/projects/rech/imi/upb99ot/hf/microsoft/phi-2"
model_name = "microsoft/phi-2"
# model_name = "gpt2"
kwargs_model = {}
llm = LLM_from_HuggingFace(
    model=model_name,
    device="cuda",
    **kwargs_model,
)

# response = llm.generate(messages=messages, n=1)
response = llm.generate(prompt="What is the capital of France?", n=1)
print(response)
# 7130Mm**71308