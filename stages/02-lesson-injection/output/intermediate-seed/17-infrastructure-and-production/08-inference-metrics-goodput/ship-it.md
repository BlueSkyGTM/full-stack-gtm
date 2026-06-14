## Ship It

**Run the benchmark against a live endpoint.** Set `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `TEST_MODEL` to a deployed endpoint — a local vLLM server, a hosted model, or an OpenAI-compatible proxy. Run the script and record TTFT, TPOT, ITL, and Goodput for all three prompt lengths. The metric that degrades most as prompt length increases tells you where your bottleneck lives: if TTFT scales linearly with prompt length, prefill is the constraint and you need a larger compute budget or a shorter prompt. If TPOT stays flat but ITL P99 spikes, the problem is decode scheduling, not prefill.

**Compute Goodput under adversarial load.** Extend the script to include deliberately overlength prompts (prompts that exceed the model's context window). Set 20 percent of requests to use a 5000-word prompt against a model with a 4096-token limit. Watch Goodput drop as those requests fail and their token budgets are wasted. The Goodput percentage is the number that determines your real cost per enriched record in production — not the throughput, not the mean latency.

**Run a sustained load test with rolling P99.** The following script runs a 2-minute sustained load test against an endpoint (or simulation), collecting E2E latency in 10-second rolling windows and reporting P99 per window. This reveals whether P99 degrades over time — a sign of KV-cache pressure building up, batch scheduler contention, or thermal throttling on the GPU:

```python
import os
import time
import random
import threading
from collections import deque

ENDPOINT = os.environ.get("OPENAI_API_BASE", "")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
DURATION_SEC = 120
WINDOW_SEC = 10
CONCURRENCY = 4

latencies = deque()
latencies_lock = threading.Lock()
stop_flag = threading.Event()

def simulated_request():
    base_ttft = random.uniform(80, 200)
    num_tokens = random.randint(80, 120)
    itls = [max(3, random.gauss(7.5, 1.5)) for _ in range(num_tokens)]
    if random.random() < 0.06:
        idx = random.randint(0, len(itls) -