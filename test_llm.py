from transformers import AutoTokenizer

from llm.llm_from_hf import LLM_from_HuggingFace
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
# assert tokenizer.chat_template is not None

# # OK to use
# prompt = tokenizer.apply_chat_template(messages, tokenize=False)
# print(prompt)


llm = LLM_from_HuggingFace(
    config={
        "model": "Qwen/Qwen3-0.6Bqqqqq",
        "device": "cuda",
        "method_truncation": "last",
        "kwargs": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": 1000,
            "do_sample": True,
        },
    }
)

# response = llm.generate(messages=messages, n=1)
response = llm.generate(prompt="What is the capital of France?", n=1)
print(response)