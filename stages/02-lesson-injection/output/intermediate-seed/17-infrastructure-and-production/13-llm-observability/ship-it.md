## Ship It

Taking this from simulation to production means three decisions. First, pick your trace backend. If you're early-stage GTM with under 100K enrichment calls per month, LangFuse's free cloud tier covers you with zero infrastructure overhead. If you're running 1M+ calls and the per-event pricing starts stinging, self-host LangFuse (it's MIT-licensed, runs on a single Postgres instance up to ~5M traces) or evaluate Arize AX if you're already on a data lake with Iceberg/Parquet tables. Second, instrument at the right layer. The cleanest approach for GTM enrichment pipelines is a decorator or context manager that wraps every LLM call — not a proxy like Helicone, because Clay and other GTM tools make their own API calls that you can't route through a proxy. Third, wire your evaluation hooks into the alert pipeline, not just the dashboard. A quality score that nobody checks is decorative.

The evaluation hook is the part most teams skip and regret. Without it, you have cost and latency visibility but zero quality visibility — you'll know that your enrichment calls cost $0.003 each and take 400ms, but you won't know that prompt version C started returning `{"icp_fit_score": null}` for 30% of accounts after a model update. The eval function doesn't need to be sophisticated. A format check (does the output parse as JSON with the expected keys?) catches most silent failures. A semantic check (does the output contain expected entities?) catches the rest. Wire either one into the trace pipeline so every call gets scored, and set an alert on the rolling average — if it drops below your threshold for 50 consecutive calls, page someone.

Here is a production-ready decorator that instruments any LLM call function with the full trace structure, computes cost, runs a quality check, and emits the trace in a format compatible with LangFuse's API.

```python
import json
import time
import uuid
from datetime import datetime, timezone
from functools import wraps

MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015}
}

_trace_buffer = []

def estimate_tokens(text):
    return max(1, len(text) // 4)

def compute_cost(model, input_tokens, output_tokens):
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    return (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])

def observe(model="gpt-4o-mini", workflow_id="default", eval_fn=None, alert_threshold=0.7):
    def decorator(fn):
        @wraps(fn)
        def wrapper(prompt, *args, **kwargs):
            trace_id = str(uuid.uuid4())
            start = time.monotonic()

            output = fn(prompt, *args, **kwargs)

            latency_ms = (time.monotonic() - start) * 1000
            input_tokens = estimate_tokens(prompt)
            output_tokens = estimate_tokens(str(output))
            cost = compute_cost(model, input_tokens, output_tokens)

            quality = eval_fn(output) if eval_fn else 1.0
            status = "OK" if quality >= alert_threshold else "REGRESSION"

            trace = {
                "id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": workflow_id,
                "model": model,
                "input": prompt,
                "output": str(output),
                "metadata": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": round(cost, 6),
                    "latency_ms": round(latency_ms, 2),
                    "quality_score": quality,
                    "status": status
                },
                "langfuse_format": True
            }

            _trace_buffer.append(trace)

            if status == "REGRESSION":
                print(f"[ALERT] Regression on {workflow_id} | trace={trace_id[:8]} | score={quality:.2f}")

            return output
        return wrapper
    return decorator

def check_icp_json(output):
    try:
        data = json.loads(output) if isinstance(output, str) else output
        return 1.0 if "icp_fit_score" in data else 0.0
    except (json.JSONDecodeError, TypeError):
        return 0.0

def check_nonempty(output):
    return 1.0 if output and len(str(output).strip()) > 0 else 0.0

@observe(model="gpt-4o", workflow_id="gtm_company_research", eval_fn=check_icp_json)
def mock_company_research(prompt):
    return json.dumps({"company": "Stripe", "icp_fit_score": 0.91, "signals": ["fintech", "series-unknown"]})

@observe(model="gpt-4o", workflow_id="gtm_company_research", eval_fn=check_icp_json)
def mock_company_research_broken(prompt):
    return json.dumps({"company": "Unknown Inc", "error": "no data found"})

@observe(model="gpt-4o-mini", workflow_id="gtm_signal_extraction", eval_fn=check_nonempty)
def mock_signal_extraction(prompt):
    return "3 signals found: Series B, fintech, 500-1000 employees"

print("=== Production-Traced GTM Enrichment Calls ===\n")

r1 = mock_company_research("Research: Stripe, fintech")
print(f"Result 1: {r1}\n")

r2 = mock_company_research_broken("Research: Unknown Inc")
print(f"Result 2: {r2}\n")

r3 = mock_signal_extraction("Extract signals: Stripe LinkedIn data")
print(f"Result 3: {r3}\n")

print("=" * 60)
print("CAPTURED TRACES (LangFuse-compatible)")
print("=" * 60)

for t in _trace_buffer:
    print(json.dumps(t, indent=2))

total_cost = sum(t["metadata"]["cost_usd"] for t in _trace_buffer)
total_tokens = sum(t["metadata"]["total_tokens"] for t in _trace_buffer)
regressions = sum(1 for t in _trace_buffer if t["metadata"]["status"] == "REGRESSION")

print("\n" + "=" * 60)
print("SHIPPING SUMMARY")
print("=" * 60)
print(f"Traces captured:  {len(_trace_buffer)}")
print(f"Total tokens:     {total_tokens}")
print(f"Total cost:       ${total_cost:.6f}")
print(f"Regressions:      {regressions}")
print(f"Trace format:     LangFuse API-compatible (POST to /api/public/traces)")
```

Run it. Three enrichment calls execute, each wrapped by the `@observe` decorator. The second call returns a broken response (no `icp_fit_score` key), which the eval function catches and flags as a regression — the alert fires inline. The trace buffer contains all three traces in LangFuse's ingestion format, ready to POST to `/api/public/traces` if you point the buffer at a real LangFuse instance instead of printing it. The shipping summary gives you the three numbers that matter: how much you spent, how many tokens you burned, and how many calls regressed on quality.

To deploy this in a real Clay + enrichment pipeline: wrap your AI enrichment functions with the decorator, set `workflow_id` to match the Clay waterfall step name, and flush `_trace_buffer` to LangFuse (or your OTel collector) on a batch schedule. Set the alert threshold based on your historical quality distribution — start at 0.7, tighten as you accumulate data.