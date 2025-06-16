import os
from typing import Dict, List
from fastapi import FastAPI, Header, HTTPException, Request
from transformers import pipeline
from llm.llm_from_hf import LLM_from_HuggingFace
from datetime import datetime, timezone
import uuid
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


model_name = "microsoft/phi-2"
kwargs_model = {}
llm = LLM_from_HuggingFace(
    model=model_name,
    device="cuda",
    **kwargs_model,
)
app = FastAPI()
auth_scheme = HTTPBearer()

API_KEY = os.getenv("TBLLM_API_KEY")
API_KEY = "1234"


# def verify_api_key(x_api_key: str):
#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Unauthorized")


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: None = Depends(verify_api_key)):
    data = await request.json()

    # Parse the model to use (for now, we only support one model)
    if "model" in data and data["model"] != model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Model {data['model']} is not supported. The model deployed right now is {model_name}.",
        )

    # Parse the messages
    messages = data.get("messages", [])
    if not messages or not isinstance(messages, list):
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing 'messages' field. Must be a [{{'role' : ..., 'content' : ...}}] list.",
        )

    # Call the Hugging Face model
    kwargs_inference = {k: v for k, v in data.items() if k not in ["model", "messages"]}
    completions, usage = llm.generate(
        messages=messages, return_usage=True, **kwargs_inference
    )

    # Build choices list with all completions
    choices = []
    for i, completion in enumerate(completions):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": completion.strip()},
                "finish_reason": "stop",
            }
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model_name,
        "choices": choices,
        "usage": usage,
    }
