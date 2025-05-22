from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tenacity import retry, wait_exponential,wait_random
import time
cfg_generation = {
    "temperature": 0.5,    
    "model": "gpt-4",
        }

@retry(wait=wait_exponential(multiplier=1, min=30, max=600)+wait_random(min=0, max=1))
def get_completion(client, prompt: str, cfg_generation, system_prompt=None, temperature=None, n=1) -> list[str]:
    """Get completion(s) from OpenAI API"""
    kwargs = cfg_generation.copy()
    if temperature is not None:
        kwargs["temperature"] = temperature
    kwargs["n"] = n
    if system_prompt is None:
        system_prompt = "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by reasoning and generating Python code."


    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
    except Exception as e:
        print("completion problem: ", e)
        too_long = "longer than the model's context length" in e.body["message"]
        if too_long:
            return [e.body["message"]] * n
        return [None] * n


        out = [choice.message.content for choice in completion.choices]
        return out
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def get_multiple_completions(client, batch_prompt: list[str], cfg_generation: dict={}, batch_tools: list[list[dict]]=None, max_workers=20, temperature=None, n=1,timeout_seconds=None)->list[list[str]]:
    """Get multiple completions from OpenAI API"""
    if isinstance(batch_prompt, str):
        batch_prompt = [batch_prompt]
    
    completions = []
    count=0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub_batch in chunks(batch_prompt, max_workers):
            
            for message in sub_batch:
                count+=1
                kwargs = {
                    "client": client,
                    "prompt": message,
                    "cfg_generation": cfg_generation,
                    "temperature": temperature,
                    "n": n
                }
                future = executor.submit(get_completion, **kwargs)
                completions.append(future)
            time.sleep(5)

            print(f"send {count} / {len(batch_prompt)} messages")

    # Retrieve the results from the futures
    if timeout_seconds is None:
        out_n = [future.result() for future in completions]
    else:
        # Retrieve the results from the futures with a timeout
        out_n = []
        for future in completions:
            try:
                result = future.result(timeout=timeout_seconds)  # Enforce timeout
                out_n.append(result)
            except TimeoutError:
                print(f"A call to get_completion timed out after {timeout_seconds} seconds")
                out_n.append(None)  # Handle timeout (e.g., append None or a default value)
            except Exception as e:
                print(f"An error occurred: {e}")
                out_n.append(None)  # Handle other exceptions
        
    return out_n
