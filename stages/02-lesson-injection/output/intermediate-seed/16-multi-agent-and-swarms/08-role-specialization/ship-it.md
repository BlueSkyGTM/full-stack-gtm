## Ship It

Production pipelines with four roles accumulate cost and latency linearly. Each role is a separate API call, and if you are processing thousands of records, the math matters. A four-role pipeline on 5,000 accounts is 20,000 API calls. At even $0.01 per call (a low estimate for a decent model), that is $200 per run. Run it daily and you are at $6,000/month just for the LLM calls — before enrichment provider costs.

Latency compounds because the roles are sequential. The Executor cannot start until the Critic approves. The Verifier cannot run until all Executor steps finish. A pipeline that takes 3 seconds per role with 3 execution steps takes roughly 3+3+9+3 = 18 seconds end-to-end. For a batch of 1,000 records processed sequentially, that is 5 hours. Parallelism helps, but rate limits constrain how many concurrent pipelines you can run.

The practical decision is when to skip roles. Here is a heuristic grounded in the distributed systems framing from Zone 16:

```python
import time
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class StageTimer:
    timings: dict = field(default_factory=dict)

    def time_stage(self, name: str, fn: Callable, *args, **kwargs):
        start = time.monotonic()
        result = fn(*args, **kwargs)
        elapsed = time.monotonic() - start
        self.timings[name] = elapsed
        return result

def should_skip_critic(stakes_score, latency_budget_ms, avg_critic_ms=3000):
    if stakes_score < 3:
        return True, "low stakes: skipping critic"
    if avg_critic_ms > latency_budget_ms * 0.4:
        return True, f"latency budget: critic would consume {avg_critic_ms}/{latency_budget_ms}ms"
    return False, "running critic"

def should_skip_verifier(stakes_score, latency_budget_ms, avg_verifier_ms=2000, schema_validated=False):
    if schema_validated and stakes_score < 4:
        return True, "schema-validated output at medium stakes: skipping LLM verifier"
    if avg_verifier_ms > latency_budget_ms * 0.3:
        return True, f"latency budget: verifier would consume {avg_verifier_ms}/{latency_budget_ms}ms"
    return False, "running verifier"

def run_pipeline_with_budget(goal, stakes_score, latency_budget_ms, timer=None):
    if timer is None:
        timer = StageTimer()

    skip_c, reason_c = should_skip_critic(stakes_score, latency_budget_ms)
    skip_v, reason_v = should_skip_verifier(stakes_score, latency_budget_ms)

    print(f"\nPipeline config for: {goal}")
    print(f"  Stakes: {stakes_score}/5, Budget: {latency_budget_ms}ms")
    print(f"  Critic: {'SKIP' if skip_c else 'RUN'} ({reason_c})")
    print(f"  Verifier: {'SKIP' if skip_v else 'RUN'} ({reason_v})")

    return {"skip_critic": skip_c, "skip_verifier": skip_v}

run_pipeline_with_budget("Draft email subject line", stakes_score=1, latency_budget_ms=5000)
run_pipeline_with_budget("Enrich lead record", stakes_score=3, latency_budget_ms=5000)
run_pipeline_with_budget("Write to CRM + trigger sequence", stakes_score=5, latency_budget_ms=15000)
```

Run this and you will see the pipeline configuration change based on stakes and latency budget. The low-stakes task skips both Critic and Verifier. The medium-stakes task runs the Critic but skips the Verifier (because the output is schema-validated). The high-stakes task runs all four roles because the cost of bad data in the CRM exceeds the cost of the extra API calls.

Monitoring matters because each role fails differently. Track per-stage failure rates independently: Planner failure (produces incomplete plans), Critic failure (approves bad plans or rejects good ones), Executor failure (API errors, rate limits, missing data), Verifier failure (false approvals or false rejections). A high Critic rejection rate means your Planner prompt is weak — fix the Planner, not the Critic. A high Verifier rejection rate means your Executor is producing incomplete artifacts — fix the Executor or add more enrichment sources.

Fallback behavior when the Verifier rejects: do not silently drop the record. Write it to a quarantine queue with the rejection reason. In the GTM context, a Verifier rejection on an enrichment record means "this account does not have enough data for confident outreach." A human SDR can review quarantined records and decide whether to proceed with incomplete data or skip the account. This is the idempotent retry pattern from distributed systems — failed records go to a dead-letter queue, not back into the main pipeline.