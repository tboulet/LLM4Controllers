import os
from typing import Dict, List
from fastapi import FastAPI, Header, HTTPException, Request
from transformers import pipeline
from llm.llm_from_hf import LLM_from_HuggingFace
from datetime import datetime, timezone
import uuid

# llm = LLM_from_HuggingFace(
#     config={
#         "model": "microsoft/phi-2",
#         "device": "cuda",
#         "method_truncation": "last",
#         "kwargs": {
#             "temperature": 0.7,
#             "top_p": 0.95,
#             "max_new_tokens": 4096,
#             "do_sample": True,
#         },
#     }
# )

app = FastAPI()
models : Dict[str, LLM_from_HuggingFace] = {}

API_KEY = os.getenv("TBLLM_API_KEY")  # Replace with your secret key or set it in environment variables
API_KEY = "1234"

def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")



@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    data = await request.json()

    # Identify the model to use
    if not "model" in data:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    model_name = data["model"]
    # Check if the model is already loaded
    if model_name not in models:
        # Load the model
        print(f"[INFO] Loading model {model_name}...")
        models[model_name] = LLM_from_HuggingFace(
            config={
                "model": model_name,
                "device": "cuda",
                "method_truncation": "last",
                "kwargs": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_new_tokens": 4096,
                    "do_sample": True,
                },
            }
        )
    
    
    # Parse OpenAI-style inputs
    messages = data.get("messages", [])
    if not messages or not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Invalid or missing 'messages'")

    # Construct prompt from messages (basic concatenation)
    prompt = ""
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role and content:
            prompt += f"{role}: {content}\n"
    prompt += "assistant:"

    # Call your Hugging Face model
    response_text = llm.generate(prompt=prompt, n=1)[0]

    # Construct OpenAI-style response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        # .now(datetime.timezone.utc
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": "microsoft/phi-2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
