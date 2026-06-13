# Shadow Traffic, Canary Rollout, and Progressive Deployment for LLMs

---

## Beat 1: Hook

Production LLM deployments fail in ways traditional services don't: semantic drift, latency spikes from longer outputs, cost explosions from prompt changes, and quality regression that status codes can't detect. A model returning HTTP 200 with confidently wrong answers is worse than a 500. This lesson covers three deployment patterns that limit blast radius when shipping LLM changes.

---

## Beat 2: Concept

**Shadow Traffic** mirrors production requests to a candidate model without serving its responses to users. Outputs are logged and compared offline. Production is unaffected; inference cost doubles.

**Canary Rollout** routes a fixed percentage of live traffic to the candidate model. Real users receive candidate responses. Blast radius is bounded by the percentage, but blast is real.

**Progressive Deployment** increments canary traffic over time (1% → 5% → 25% → 100%), with automated rollback triggered by guardrail violations. This is the full lifecycle pattern; shadow and canary are two points on its spectrum.

Key distinction: shadow traffic observes without risk; canary accepts bounded risk for faster signal; progressive deployment automates the escalation decision.

---

## Beat 3: Mechanism

### Traffic Mirroring (Shadow)

1. Intercept request at the gateway or application layer.
2. Clone the request payload.
3. Send clone to candidate model endpoint asynchronously.
4. Log candidate response alongside production response.
5. Offline, run comparison: exact match (classification), semantic similarity (generation), latency delta, cost delta.

Mechanism is a sidecar proxy or middleware that duplicates the request. The comparison function is domain-specific — there is no generic "LLM output diff."

### Canary Routing

1. Assign a traffic-splitting key per request (user ID hash, random uniform).
2. Route to candidate model if key falls within canary percentage.
3. Emit metrics per model version: latency p50/p95/p99, error rate, guardrail violation rate, cost per request.
4. Compare metrics against baseline thresholds.
5. Roll back if thresholds are breached within the observation window.

Routing is implemented via load-balancer weights (Envoy, Nginx) or application-level branching. The guardrail check is the critical piece — without it, canary is just hoping.

### Progressive Rollout Automation

1. Define stages: percentages and minimum dwell time per stage.
2. Define promotion criteria: guardrail pass rate ≥ X%, latency p99 ≤ Yms, cost within Z%.
3. Define rollback criteria: any hard threshold breach.
4. Controller promotes, holds, or rolls back based on accumulated metrics.
5. Runbook for manual override when automated decisions are wrong.

This is a state machine. Stages are states. Transitions are driven by metric evaluation. The controller is either a custom orchestrator or a platform feature (Litellm proxy, Argo Rollouts with custom metrics).

### LLM-Specific Complications

- **Non-deterministic outputs**: same input, different output each call. Comparison must be statistical, not diff-based.
- **Latency variance**: output token count varies, so latency distribution is bimodal. p99 is the wrong metric; use per-token latency or time-to-first-token.
- **Quality evaluation**: automated evals (LLM-as-judge, embedding cosine similarity) are proxies. Ground truth requires human review or gold datasets.
- **Cost tracking**: shadow traffic incurs full inference cost with zero revenue. Budget caps are mandatory.

---

## Beat 4: Code

### Exercise Hook: Easy — Shadow Traffic Logger

Build a middleware that intercepts requests to a production LLM endpoint, clones them to a candidate endpoint, and logs both responses with timestamps and latency. Print a comparison summary.

### Exercise Hook: Medium — Canary Router with Guardrails

Implement a request router that splits traffic 90/10 between production and candidate models. Inject a synthetic regression (candidate returns truncated output). Detect the regression automatically via a simple guardrail (output length check) and print a rollback recommendation.

### Exercise Hook: Hard — Progressive Deployment Controller

Build a state machine that progresses through stages [1%, 5%, 25%, 100%] with configurable dwell times. Simulate requests, accumulate metrics, and auto-promote or auto-rollback based on guardrail violation rate. Print the full state transition log.

---

## Beat 5: Use It

**GTM Redirect — Zone 2 (Enrichment) and Zone 3 (Orchestration)**

When deploying LLM-powered enrichment in Clay — scoring leads, classifying companies, generating personalization — the enrichment prompt is production software. A prompt change that degrades classification accuracy silently poisons the downstream CRM.

Shadow traffic pattern: run the new enrichment prompt alongside the old one on a sample of recent accounts. Compare outputs. Check if company classifications shifted. Check if personalization quality dropped. This is the Clay waterfall's validation step before swapping a column formula.

Canary pattern: route 5% of new enrichment runs through the candidate prompt. Monitor enrichment failure rate in Clay's run history. If failures spike, revert the column formula. If stable, increase.

Progressive deployment: for high-volume orchestration workflows (agent sequences, multi-step outreach), roll out new agent logic to a single IC persona first, then expand. The "stage" is the audience segment, not a traffic percentage. Same mechanism, different split key.

[CITATION NEEDED — concept: Clay waterfall validation step for prompt changes]

---

## Beat 6: Ship It

**Deployment checklist for LLM model or prompt changes:**

1. Define the comparison function before deploying. "We'll know it when we see it" is not a comparison function. Write it down: metric, threshold, evaluation method.
2. Run shadow traffic for at least N requests (N = enough for statistical significance on your key metric; typically 500–1000 for classification, more for generative tasks).
3. Review shadow results. If automated evals pass, spot-check 20–50 outputs manually. Do not skip this step.
4. Set guardrail thresholds for canary: latency ceiling, cost ceiling, output-length floor, format-compliance rate. Hard thresholds, not vibes.
5. Start canary at 1–5%. Dwell time minimum: long enough to see edge cases. For B2B GTM workflows processing 100s of accounts/day, this may be 24–48 hours.
6. Monitor. Promote or rollback. No manual overrides without a written reason in the deploy log.
7. At 100%, keep dual-logging for one full cycle before deprecating the old model.

**Observability requirements:** per-model-version metrics for latency, cost, guardrail pass rate, and a sampled output log for human review. Without all four, you are deploying blind.