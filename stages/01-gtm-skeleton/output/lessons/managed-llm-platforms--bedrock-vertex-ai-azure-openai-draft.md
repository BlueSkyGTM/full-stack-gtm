# Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI

---

## Learning Objectives

1. Configure authenticated API calls to Bedrock, Vertex AI, and Azure OpenAI
2. Compare pricing models, request schemas, and quota systems across the three providers
3. Implement a multi-provider routing function that falls back across platforms on failure
4. Evaluate provisioned-throughput vs. on-demand tradeoffs for high-volume inference
5. Diagnose authentication, quota, and latency differences from API response metadata

---

## Beat 1: Hook — "The Multi-Provider Problem"

You will ship a prompt that works. Then your single-provider deployment will throttle, deprecate a model version, or fail a compliance review. Multi-provider routing is not a nice-to-have; it is a production requirement. This lesson covers the three managed platforms that wrap foundation models behind unified APIs, and how to abstract over them so your inference layer survives provider-specific failures.

---

## Beat 2: Concept — "Abstraction, Authentication, and Quota"

The mechanism: each managed platform provides a thin API surface over hosted model weights. The differences that matter are authentication flow (IAM vs. service accounts vs. API keys), request body schema (each provider serializes differently), and quota semantics (tokens-per-minute, requests-per-minute, provisioned throughput units). Walk through each provider's identity model, request format, and pricing granularity. Show where they converge (OpenAI-compatible endpoints on Azure, model families across Bedrock) and where they diverge (Vertex AI's endpoint-per-model pattern, Bedrock's cross-region inference).

Exercise hooks:
- **Easy:** Map each provider to its authentication pattern in a comparison table
- **Medium:** Decode a real Bedrock request body and identify model-specific parameters
- **Hard:** Calculate cost-per-1K-tokens across all three providers for a 4K-input / 500-output payload and identify the cheapest option

---

## Beat 3: Demo — "Three Providers, Same Prompt, Same Output"

Write a Python script that sends an identical prompt to Bedrock (Claude), Vertex AI (Gemini), and Azure OpenAI (GPT-4o). Print each response alongside latency and token usage metadata. Use environment variables for credentials. Show where request serialization differs and where response shapes diverge. All code runs from the terminal with observable print output.

```python
import os
import json
import time
import httpx

prompt = "Extract the company name and industry from: 'Acme Corp, a fintech startup, raised $20M.'"

results = {}

# Azure OpenAI
start = time.time()
resp = httpx.post(
    f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/deployments/{os.environ['AZURE_OPENAI_DEPLOYMENT']}/chat/completions?api-version=2024-06-01",
    headers={"api-key": os.environ["AZURE_OPENAI_KEY"]},
    json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 100},
    timeout=30
)
elapsed = time.time() - start
data = resp.json()
results["azure_openai"] = {
    "status": resp.status_code,
    "latency_ms": round(elapsed * 1000),
    "content": data["choices"][0]["message"]["content"],
    "usage": data.get("usage", {})
}

for provider, r in results.items():
    print(f"\n=== {provider} ===")
    print(f"Status: {r['status']}")
    print(f"Latency: {r['latency_ms']}ms")
    print(f"Response: {r['content']}")
    print(f"Tokens: {r.get('usage', {})}")
```

Exercise hooks:
- **Easy:** Run the script with one provider and confirm output
- **Medium:** Extend the script to include a Bedrock call using boto3
- **Hard:** Add retry logic with exponential backoff and log which provider succeeds on retry

---

## Beat 4: Use It — "Enrichment at Scale with Provider Fallback"

GTM redirect: this is the inference backbone for **Zone 2 (Signal)** enrichment workflows — specifically company research, signal extraction, and personalization pipelines in the GTM topic map. When you enrich 10K accounts with LLM-extracted signals, you need multi-provider routing to handle quota limits and cost optimization.

Show a routing function that tries providers in order of cost, falls back on 429/5xx, and logs which provider served each request. This is the same pattern used in Clay waterfalls for enrichment, applied at the LLM provider layer.

Exercise hooks:
- **Easy:** Add a second provider to the demo script with a simple if/else fallback
- **Medium:** Build a routing function that tries Azure OpenAI first, falls back to Bedrock on failure, and logs provider usage counts
- **Hard:** Implement a cost-aware router that selects provider based on input token count thresholds derived from each provider's pricing tiers

---

## Beat 5: Ship It — "Production Inference Layer"

Write a production-ready inference module with provider abstraction, structured error handling, and observability. Include credential validation on startup, per-provider circuit breakers that trip after N consecutive failures, and a `/health` check that confirms each provider is reachable. Output a JSON status report to stdout.

```python
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ProviderHealth:
    name: str
    consecutive_failures: int = 0
    last_success: Optional[float] = None
    circuit_open: bool = False
    total_requests: int = 0
    total_failures: int = 0

THRESHOLD = 3
providers = {
    "azure": ProviderHealth("azure_openai"),
    "bedrock": ProviderHealth("bedrock"),
}

def check_circuit(name: str) -> bool:
    p = providers[name]
    if p.circuit_open and p.last_success and (time.time() - p.last_success > 60):
        p.circuit_open = False
        p.consecutive_failures = 0
    return not p.circuit_open

def record_result(name: str, success: bool):
    p = providers[name]
    p.total_requests += 1
    if success:
        p.consecutive_failures = 0
        p.last_success = time.time()
    else:
        p.consecutive_failures += 1
        p.total_failures += 1
        if p.consecutive_failures >= THRESHOLD:
            p.circuit_open = True

print(json.dumps({k: {"requests": v.total_requests, "failures": v.total_failures, "circuit_open": v.circuit_open} for k, v in providers.items()}, indent=2))
```

Exercise hooks:
- **Easy:** Run the health check output and confirm circuit breaker state
- **Medium:** Add a Vertex AI provider to the circuit breaker and test failover
- **Hard:** Add a Prometheus-compatible metrics endpoint that exposes per-provider latency percentiles and error rates

---

## Beat 6: Spot Check

Assessment hooks only — these map directly to the learning objectives:

1. Given three API error logs, identify which provider returned each and diagnose the failure type (auth vs. quota vs. model-not-found)
2. A prompt works on Azure OpenAI but returns a different schema error on Bedrock — diagnose the request body difference
3. Your team runs 500K enrichment requests/month — calculate cost across providers and recommend a routing strategy
4. A circuit breaker trips on your primary provider — trace the logic that determines when it reopens and what happens to in-flight requests

---

## GTM Redirect Rules

- **Use It** maps to: Zone 2 (Signal) — enrichment waterfall inference layer. This is the provider-routing backbone behind company research, signal extraction, and personalization at scale.
- **Ship It** maps to: foundational for all GTM zones. Multi-provider LLM routing is infrastructure, not a GTM-specific tactic. The redirect is "foundational for Zone 1–3 inference workflows" — not a fabricated application to a single GTM step.