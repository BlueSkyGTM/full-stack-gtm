# Chaos Engineering for LLM Production

## Learning Objectives

1. Inject controlled failures into LLM API calls and capture system response metrics
2. Build a chaos test harness that simulates rate limits, timeouts, and malformed responses
3. Configure retry and fallback strategies that survive specific failure modes
4. Evaluate system resilience under degraded conditions using defined pass/fail criteria
5. Implement a monitoring baseline that detects degradation from chaos experiment results

## The Problem

You shipped an LLM pipeline. It works on a clean test set. The latency looks fine in staging. Then production happens.

An upstream provider throttles you mid-run — your batch enrichment of 5,000 company records hits a 429 on row 847, and the error cascades into every downstream column that depends on that enrichment step. A context window shrinks after a model update that the vendor announced in a changelog nobody on your team read. A structured output call that was supposed to return `{"icp_fit_score": 0.85}` comes back as `"I cannot assist with that request."` — raw text, no JSON — and your parsing layer throws an exception that kills the worker process.

These are not edge cases. They are the default operating conditions for LLM production systems. The question is not whether they will happen — it is whether your system degrades gracefully when they do.

Traditional chaos engineering (originating with Netflix's Chaos Monkey in 2011, later formalized through CNCF projects like LitmusChaos and Chaos Mesh) developed discipline around this for infrastructure: kill a pod, sever a network link, saturate disk I/O, observe whether the system self-heals. LLM chaos engineering inherits the same epistemology — provoke the failure on purpose, in a controlled scope, before your customers provoke it for you — but adds failure classes that infrastructure chaos never had to consider. A model that refuses to answer is not an infrastructure fault. A response that violates your output schema is not a network timeout. These are semantic failures, and they require their own experiment design.

## The Concept

The core mechanism is a hypothesis test. You define a **steady state** — the normal behavior your