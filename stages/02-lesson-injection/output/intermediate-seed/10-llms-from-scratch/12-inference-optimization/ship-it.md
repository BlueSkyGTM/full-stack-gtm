## Ship It

Deploy a quantized model behind an OpenAI-compatible endpoint using vLLM. This gives you a local server that accepts the same request format as the OpenAI API, so your existing pipeline code works without modification.

First, create the serving configuration:

```python
import subprocess
import time
import requests
import json
import concurrent.futures
import statistics

SERVER_CMD = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--quantization", "awq",
    "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.85",
    "--max-num-seqs", "64",
    "--port", "8000",
]

print("Starting vLLM server...")
server_process = subprocess.Popen(
    SERVER_CMD,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

print("Waiting for server to be ready...")
for attempt in range(120):
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            print(f"Server ready after {attempt + 1} attempts")
            break
    except requests.ConnectionError:
        pass
    time.sleep(2)
else:
    print("Server failed to start within timeout")
    server_process.terminate()
    exit(1)

PROMPT = "Write a two-sentence product description for a CRM tool."
TOKENS_PER_RESPONSE = 50