# Chaos Engineering for LLM Production

## Learning Objectives

1. Inject controlled failures into LLM API calls and capture system response
2. Build a chaos test harness that simulates rate limits, timeouts, and malformed responses
3. Configure retry and fallback strategies that survive specific failure modes
4. Evaluate system resilience under degraded conditions using defined pass/fail criteria
5. Implement a monitoring baseline that detects degradation from chaos test results

---

## Beat 1: Hook

You shipped an LLM pipeline. It works on a clean test set. Now an upstream provider throttles you mid-run, a context window shrinks after a model update, and a structured output call returns raw text with no JSON. Chaos engineering is the discipline of provoking these failures on purpose, before your prospects provoke them for you.

---

## Beat 2: Concept

The core mechanism: define a "steady state" (your pipeline's normal output distribution), then inject a specific failure hypothesis (e.g., "if the model returns a 429, the fallback model receives the original prompt within 2 seconds"). The blast radius is the scope of the experiment — one API call, one enrichment pipeline, or one full workflow. Cover the four failure classes most relevant to LLM production: infrastructure (timeouts, rate limits), model behavior (refusals, format violations), data (truncated context, encoding errors), and dependency (embedding API down, vector store stale).

---

## Beat 3: Demonstration

Build a minimal chaos harness in Python. Define a steady-state function that calls an LLM, a failure injector that wraps the call with a simulated 429 or timeout, and an assertion that checks whether the fallback executed. Print the experiment result (pass/fail), the latency delta, and the fallback model's output. All output observable in terminal. Tool mention: Litmus Chaos implements this pattern for Kubernetes; this is the same pattern, scoped to an LLM call.

---

## Beat 4: Guided Exercise

**Easy:** Modify the demonstrated harness to inject a new failure mode — a simulated refusal response (e.g., "I cannot assist with that") — and assert that the fallback model receives the prompt.

**Medium:** Build a multi-failure sweep: run the pipeline against all four failure classes in sequence, log pass/fail for each, and print a summary table.

**Hard:** Add latency tracking and assert that the total fallback chain completes under a defined SLA threshold. Flag any failure class that violates it.

---

## Beat 5: Use It

GTM redirect: **Zone 2 — Data & Enrichment, Clay waterfall pattern.** Clay implements a waterfall enrichment flow that calls LLMs (and other APIs) for each row. When you chaos-test your Clay workflows, you are testing whether the waterfall degrades gracefully — does a rate limit on the first LLM step cascade and corrupt downstream enrichment columns, or does Clay's retry logic contain the blast radius? Build a chaos test that simulates a 429 on the first enrichment step and verify whether subsequent columns still populate from the fallback or remain stale.

---

## Beat 6: Ship It

Deploy a chaos experiment as a CI step. The experiment runs against a staging LLM endpoint, injects one failure class per run, and fails the build if the fallback chain does not execute or exceeds the SLA threshold. Log results to a time-series store. Production relevance: this is how you catch silent degradation when a provider updates a model and your structured output parsing breaks. [CITATION NEEDED — concept: chaos engineering adoption metrics for LLM production systems]

---

## GTM Redirect Rules

- **Zone 2 — Data & Enrichment / Clay Waterfall:** The primary redirect. Clay's enrichment waterfall is an LLM pipeline with implicit fallback logic. Chaos engineering tests whether that logic holds under failure.
- **Foundational for Zone 3 (Outbound):** If an outbound workflow generates personalized emails via LLM, chaos testing verifies that a model timeout does not send a partially rendered template.
- No fabricated connections to Zones where LLM failure behavior is irrelevant.

---

## Code Rules Reminder

- All demonstration code runs unmodified in Claude Code Desktop (terminal)
- Every code block prints observable output confirming the mechanism worked
- No scaffolded/fill-in-the-blank code in exercises — exercises describe what to build, not partial code