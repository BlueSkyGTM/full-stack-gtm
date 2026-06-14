## Ship It

The token-budget invariant that lets LLaVA-OneVision handle all three modalities also creates a specific observability requirement: **visual-token-budget drift is your degradation signal**. When screenshots arrive at unexpected resolutions, when video frame sampling produces more frames than the budget anticipates, or when the projection layer starts receiving tokens outside its training distribution, output quality degrades silently. The model still returns text — it just returns worse text. Zone 12 observability (tracing, logging, and feedback-loop monitoring) is how you catch this before your enrichment table fills with garbage.

Instrument every inference call with four traces: `modality`, `num_visual_tokens_allocated`, `num_visual_tokens_actual`, and `response_length`. When `num_visual_tokens_actual` consistently exceeds `num_visual_tokens_allocated` by more than 20%, your input pipeline is producing images or videos at higher resolution than the budget expects. This is the visual equivalent of reply-rate drift in an email sequence — the system is still running, but the quality is eroding.

```python
import json
import time
from datetime import datetime

class VisualEnrichmentTracer:
    def __init__(self, log_path="/tmp/visual_enrichment_trace.jsonl"):
        self.log_path = log_path
        self.budget_threshold = SINGLE_IMAGE_BUDGET * 1.2
    
    def trace(self, account_id, modality, num_inputs, tokens_per_input, response, prompt_hash):
        total_tokens = num_inputs * tokens_per_input
        over_budget = total_tokens > self.budget_threshold
        short_response = len(response) < 20
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "account_id": account_id,
            "modality": modality,
            "num_inputs": num_inputs,
            "tokens_per_input": tokens_per_input,
            "total_visual_tokens": total_tokens,
            "over_budget": over_budget,
            "response_length": len(response),
            "short_response_flag": short_response,
            "prompt_hash": prompt_hash,
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        return entry
    
    def health_check(self):
        entries = []
        try:
            with open(self.log_path) as f:
                entries = [json.loads(line) for line in f]
        except FileNotFoundError:
            return {"status": "no_traces_yet", "entries": 0}
        
        total = len(entries)
        over = sum(1 for e in entries if e["over_budget"])
        short = sum(1 for e in entries if e["short_response_flag"])
        
        return {
            "total_calls": total,
            "over_budget_calls": over,
            "over_budget_rate": f"{over/total*100:.1f}%" if total else "N/A",
            "short_response_calls": short,
            "short_response_rate": f"{short/total*100:.1f}%" if total else "N/A",
            "status": "degraded" if over/total > 0.15 if total else False else "healthy",
        }

tracer = VisualEnrichmentTracer()

simulated_calls = [
    ("acme-corp", "single", 1, 2880, "Acme Corp provides cloud infrastructure monitoring with usage-based pricing starting at $49/mo.", "hash_001"),
    ("beta-inc", "multi", 4, 576, "Beta Inc shifted from developer-focused to enterprise positioning between Q2 and Q4.", "hash_002"),
    ("gamma-llc", "video", 16, 196, "", "hash_003"),
    ("delta-co", "multi", 8, 576, "Delta Co added a pricing page and removed the 'Request a Demo' CTA.", "hash_004"),
    ("epsilon-io", "single", 1, 3100, "Epsilon IO uses Snowflake and dbt, pricing starts at $2,000/mo enterprise tier.", "hash_005"),
]

for call in simulated_calls:
    entry = tracer.trace(*call)
    flags = []
    if entry["over_budget"]:
        flags.append("OVER_BUDGET")
    if entry["short_response_flag"]:
        flags.append("SHORT_RESPONSE")
    flag_str = f" [{', '.join(flags)}]" if flags else ""
    print(f"{entry['account_id']:<12} {entry['modality']:<8} tokens={entry['total_visual_tokens']:>5} resp_len={entry['response_length']:>3}{flag_str}")

health = tracer.health_check()
print(f"\nPipeline Health: {health['status']}")
print(f"  Total calls:        {health['total_calls']}")
print(f"  Over-budget rate:   {health['over_budget_rate']}")
print(f"  Short-response rate:{health['short_response_rate']}")
```

The tracer above shows two degradation signals: `over_budget` flags inputs where the actual token count exceeds the budget threshold (Epsilon-IO's screenshot arrived at 3100 tokens instead of 2880 — likely a higher-resolution capture than expected), and `short_response_flag` catches empty or near-empty outputs (Gamma-LLC's video produced no response — likely a frame-sampling failure that produced zero decodable frames). In a production enrichment pipeline, either signal should trigger an alert. The over-budget rate exceeding 15% means your screenshot crawler is capturing images at resolutions the projection layer was not trained on. The short-response rate exceeding 5% means either your frame sampler is broken or the model is receiving corrupted inputs.

This tracing setup monitors your visual-enrichment pipeline in real time; token-budget drift and response-length collapse are your model-degradation signals, exactly as reply-rate drift signals sequence fatigue in outbound campaigns. Wire the tracer output into whatever dashboard your team already uses for pipeline health — Datadog, Grafana, or even a Slack webhook that posts when degradation rate crosses threshold. The point is not to build new observability infrastructure. The point is to extend your existing Zone 12 tracing to cover the visual modality that your text-only enrichment never needed.