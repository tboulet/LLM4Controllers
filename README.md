# LLM 4 Controllers

This project aims to study the use of large language models (LLMs) to generate controllers for agents in reinforcement learning tasks. We use a LLM-based Hierarchical Controller Generation (HCG) agent to interact with a task-based environment.

# Installation

Clone the repository and create a virtual environment.
The repo work (at least) in python 3.10.

```bash
git clone git@github.com:tboulet/LLM4Controllers.git
cd LLM4Controllers
python -m venv venv
source venv/bin/activate   # on Windows, use `venv\Scripts\activate.bat`
```

### Install the requirements

Install torch (command may vary depending on your system and CUDA version), and then install the requirements from the `requirements.txt` file.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

# Usage

To run any experiment, run the ``run.py`` file with the desired configuration. We use Hydra as our configuration manager. Configs are stored in the ``configs`` folder.

```bash
python run.py
```

To use a specific configuration, use the ``--config-name`` argument.

```bash
python run.py --config-name=my_config
```
 
You can modify a specific argument thanks to Hydra's override system.

```bash
python run.py path.to.arg.in.config=new_value
```

In particular, the 3 main components interacting with each other in the experiments are the agent (HCG by default), the task-based environment and the LLM that the agent will use (VLLM pipeline by default).

```bash
python run.py agent=hcg env=minigrid llm=vllm llm.model=Qwen/Qwen2.5-1.5B-Instruct
```

# Logging

The results of the experiments are stored in the ``logs`` folder. Each run is stored in a separate folder with the name of the configuration used, and is also stored in ``logs/_last``. Each log folder contains ``task_x`` folders, where ``x`` is the task number. This folder contains the received prompt, agent answer, controller generated, error traces, and videos of the episode.

# Hosting LLMs with TBLLM

TBLLM is a functionnal but non-optimized equivalent of VLLM. You can host LLMs on any machine connected to internet, and then you can call them using the OpenAI API.

To host an LLM (currently only microsoft/phi-2 will be used), run the following command:

```bash
uvicorn tbllm.run:app --host 0.0.0.0 --port 8000
```

Then, you can use the LLM using the OpenAI API

```python
from openai import OpenAI

host_name = "localhost"  # or the IP address of the machine hosting the LLM
client = OpenAI(
    api_key="1234",
    base_url = f"http://{host_name}:8000/v1",
)

response = client.chat.completions.create(
    model="microsoft/phi-2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.3,
    n=2,
)

print(response)
```

