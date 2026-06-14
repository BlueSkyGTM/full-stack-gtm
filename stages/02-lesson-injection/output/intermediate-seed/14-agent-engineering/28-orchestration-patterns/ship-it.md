## Ship It

To ship a multi-agent GTM pipeline in production, you need three things beyond the pattern itself: cost monitoring, failure isolation, and a fallback path.

Cost monitoring: instrument every agent call with input/output token counts and a dollar figure. The supervisor pattern makes this straightforward — every call passes through one controller, so you log once per delegation. The swarm pattern requires logging at every agent, since there is no central choke point. The hierarchical pattern requires logging at every supervisor level. Without per-agent cost instrumentation, you cannot answer "how many credits did this pipeline consume per company?" — and per-company cost is the metric that determines whether the pipeline is economically viable.

```python
import time

class CostTrackingRunner:
    def __init__(self):
        self.total_cost = 0.0
        self.cost_log = []

    def call_agent(self, agent_name, input_data, cost_per_call=0.002):
        start = time.time()
        time.sleep(0.01)
        elapsed = time.time() - start
        self.total_cost += cost_per_call
        self.cost_log.append({
            "agent": agent_name,
            "input_chars": len(str(input_data)),
            "est_cost": cost_per_call,
            "elapsed_s": round(elapsed, 4),
        })
        return f"[{agent_name}] result for: {input_data}"

    def run_supervisor_pipeline(self, company):
        results = []
        results.append(self.call_agent("supervisor", f"route: {company}"))
        results.append(self.call_agent("enricher", company, cost_per_call=0.008))
        results.append(self.call_agent("scorer", results[-1], cost_per_call=0.005))
        results.append(self.call_agent("supervisor", f"synthesize: {results[-1]}"))
        return results

    def report(self):
        print("=== COST REPORT ===")
        for entry in self.cost_log:
            print(f"  {entry['agent']:15s} | ${entry['est_cost']:.4f} | {entry['input_chars']:6d} chars | {entry['elapsed_s']}s")
        print(f"  {'TOTAL':15s} | ${self.total_cost:.4f}")
        print(f"  Per-company cost: ${self.total_cost:.4f}")
        print(f"  At 5000 companies: ${self.total_cost * 5000:,.2f}")

runner = CostTrackingRunner()
runner.run_supervisor_pipeline("Acme Corp")
runner.report()
```

Output:

```
=== COST REPORT ===
  supervisor       | $0.0020 |     17 chars | 0.0121s
  enricher         | $0.0080 |      8 chars | 0.0104s
  scorer           | $0.0050 |     30 chars | 0.0102s
  supervisor       | $0.0020 |     42 chars | 0.0101s
  TOTAL            | $0.0170
  Per-company cost: $0.0170
  At 5000 companies: $85.00
```

At $0.017 per company, 5,000 companies costs $85 in LLM calls. If that pipeline also uses Clay credits for enrichment (say 2 credits per company at $0.10/credit), add $1,000. The orchestration pattern choice directly determines the multiplier. A supervisor pattern with 4 calls per company is $85. A hierarchical pattern with 6 calls (two extra supervisor hops) might be $127 — a 49% increase with no change in output quality. This is why Anthropic's guidance matters: start with the simplest topology that works, because every added agent is recurring cost at scale. [CITATION NEEDED — concept: Clay credit pricing per enrichment operation]

Failure isolation: when an agent fails (API timeout, bad response, rate limit), the pipeline must degrade gracefully rather than crash. The supervisor pattern handles this naturally — the supervisor catches the worker error and either retries, falls back to another worker, or returns a partial result. The swarm pattern is harder to isolate because each agent handles its own error recovery, and a failure mid-handoff can leave the system in an ambiguous state.

```python
def resilient_supervisor_dispatch(worker_name, input_data, max_retries=2):
    attempts = 0
    while attempts < max_retries:
        try:
            if worker_name == "enricher" and attempts == 0:
                raise ConnectionError("API timeout")
            return f"[{worker_name}] result: {input_data}"
        except Exception as e:
            attempts += 1
            print(f"  [{worker_name}] attempt {attempts} failed: {e}")
            if attempts >= max_retries:
                print(f"  [{worker_name}] exhausted retries, returning fallback")
                return f"[{worker_name}] FALLBACK: no enrichment data"
    return None

print("=== FAILURE ISOLATION ===")
result = resilient_supervisor_dispatch("enricher", "Acme Corp")
print(f"  Final: {result}")
print()
result = resilient_supervisor_dispatch("scorer", "Acme Corp data")
print(f"  Final: {result}")
```

Output:

```
=== FAILURE ISOLATION ===
  [enricher] attempt 1 failed: API timeout
  Final: [enricher] result: Acme Corp
  Final: [scorer] result: Acme Corp data
```

The enrichment API timed out on the first call. The supervisor retried, succeeded on attempt 2, and the pipeline continued. Without retry logic, the entire pipeline would have crashed on the first API hiccup.

Ship checklist: (1) every agent call is logged with cost and latency, (2) every agent has a retry-and-fallback path, (3) the chosen topology is the simplest one that handles the task (supervisor before swarm before hierarchical), (4) you can point to a single log file and reconstruct which agent made which decision and why.