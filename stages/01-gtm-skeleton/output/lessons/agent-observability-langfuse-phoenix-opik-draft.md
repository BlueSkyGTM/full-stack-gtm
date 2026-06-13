# Agent Observability: Langfuse, Phoenix, Opik

## Beat 1: Hook — Why Your Agent Is a Black Box

You shipped an agent. It works. Then it doesn't. You have no idea why. That gap between "it worked in testing" and "it's failing in production" is where observability lives — or doesn't.

---

## Beat 2: Concept — Traces, Spans, and the Three-Layer Model

Introduce the mechanism: every agent run produces a trace (the full execution), spans (individual LLM calls, tool uses, retrievals), and metadata (tokens, latency, cost, model version). Explain the instrumentation pattern — decorator-based wrapping vs. callback-based integration vs. proxy interception — before naming any tool. Then position Langfuse (open-source, decorator-first), Phoenix (Arize, local-first with embedded Arize backend), and Opik (Comet, eval-focused observability) as three implementations of the same trace-span-metadata model.

---

## Beat 3: Demonstration — Instrument a Multi-Step Agent and Inspect the Output

Show a minimal ReAct-style agent that makes 2-3 LLM calls and 1 tool call. Instrument it with Langfuse's decorator approach. Run it. Print the trace URL and confirm token counts, latency per span, and total cost are captured. Then show the same agent instrumented with Phoenix's session-based approach and compare the output structure.

**Exercise hooks:**
- *Easy:* Add a Langfuse decorator to a single LLM call and print the trace ID.
- *Medium:* Instrument a 2-step agent (LLM → tool → LLM) and confirm spans appear in the correct parent-child hierarchy.
- *Hard:* Run the same prompt through both Langfuse and Phoenix instrumentation, then write a comparison function that asserts both captured the same token count (±5% tolerance).

---

## Beat 4: Use It — GTM Redirect: Agent Performance as Revenue Signal

Map to the GTM cluster for automated outbound/inbound agents. The redirect: if your outbound agent generates personalized emails via a 3-step chain (research → draft → refine), observability tells you which step fails, how much each email costs, and whether the "refine" step actually improves response rates or just burns tokens. [CITATION NEEDED — concept: mapping agent trace data to GTM funnel metrics like cost-per-meeting-booked].

**Exercise hooks:**
- *Easy:* Tag your Langfuse traces with a `campaign_id` metadata field and filter by it in the dashboard.
- *Medium:* Write a function that pulls the last 50 traces for a given campaign and computes average cost-per-email and p95 latency.
- *Hard:* Correlate trace-level "refine step was called" data with downstream reply rates (mock data provided) and determine if the refine step has positive ROI.

---

## Beat 5: Ship It — Production Observability Pipeline

Cover the deployment pattern: local development with Phoenix (zero-config, embedded SQLite), staging with Langfuse self-hosted (Docker Compose with Postgres), and production routing to whichever backend matches your compliance requirements. Address retention policies, PII scrubbing in trace payloads, and the sampling decision (log everything vs. log failures + 10% sample).

**Exercise hooks:**
- *Easy:* Spin up Langfuse locally via Docker and confirm you can POST a trace to it.
- *Medium:* Write a middleware function that strips email addresses from span inputs before they reach the observability backend.
- *Hard:* Implement a sampling function that logs 100% of traces where `output_quality_score < 0.7` and 5% of all others, then verify the sampling ratio in your local dashboard.

---

## Beat 6: Extend It — Evaluations, Custom Scorers, and Alerting

Point to the next layer: once you have traces, you can score them automatically — toxicity, hallucination, relevance — and feed those scores back into your dashboards. Langfuse has a scoring API; Phoenix integrates with Arize's evaluation suite. Mention that alerting on metric thresholds (cost spike, latency regression, score degradation) closes the loop from observability to incident response. Cross-reference the Evals lesson for scorer implementation details.

**Exercise hooks:**
- *Easy:* Attach a manual score to a Langfuse trace via the API and confirm it appears in the UI.
- *Medium:* Write an async scorer that calls a second LLM to grade the first LLM's output, then attaches that score to the original trace.
- *Hard:* Set up a scheduled job (cron or GitHub Action) that pulls the last hour of traces, computes average hallucination score, and fires a webhook if it exceeds 0.3.