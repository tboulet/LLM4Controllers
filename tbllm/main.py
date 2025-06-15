# main.py
from fastapi import FastAPI, Header, HTTPException, Request
from transformers import pipeline
from llm.llm_from_hf import LLM_from_HuggingFace


llm = LLM_from_HuggingFace(
    config={
        "model": "microsoft/phi-2",
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

app = FastAPI()

API_KEY = "1234"  # Replace with your secret key
def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/predict/")
async def predict(request: Request, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    data = await request.json()
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' field")
    result = llm.generate(prompt=text, n=1)
    return {"result": result}
