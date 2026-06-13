# LLM Observability Stack Selection

## Learning Objectives

1. Compare observability tools by evaluating their tracing, logging, and cost-tracking mechanisms against production requirements.
2. Configure a trace pipeline that captures prompt inputs, model outputs, token counts, and latency per request.
3. Implement per-request cost attribution tied to workflow or campaign.
4. Detect output quality regressions using structured evaluation hooks in the observability loop.
5. Evaluate vendor and self-hosted observability stacks for debug depth, retention, and cost tradeoffs.

---

## Beat 1: Hook

**The silent failure problem.** LLM calls fail silently—wrong outputs, runaway token costs, 10-second latencies—and you won't know until a human complains or a bill arrives. Observability is the difference between "something feels off" and "prompt variant B on the enrichment waterfall started returning empty objects at 2:14 PM." The stack you pick determines whether you find that in minutes or weeks.

---

## Beat 2: Concept

**Three mechanisms, one stack.** Observability for LLM systems requires three distinct mechanisms: (1) **request tracing**—capturing the full prompt→output pair plus metadata for every call, (2) **metric aggregation**—token counts, latencies, error rates rolled up by model, prompt version, and workflow, and (3) **evaluation hooks**—automated checks that score outputs and flag regressions. Traditional APM (Application Performance Monitoring) handles #2 but not #1 or #3. LLM-native tools (LangSmith, LangFuse, Helicone, Arize) add prompt/output capture and eval loops. Self-hosted options (OpenTelemetry + PostHog, or a custom logs table) give control but require assembly. The selection matrix comes down to: trace granularity, retention period, eval integration, and per-event cost at your call volume.

**Key tradeoff:** Managed tools charge per trace. At 1M enrichment calls/month, that's a line-item decision. Self-hosted shifts cost to infrastructure and engineering time.

---

## Beat 3: Demo

Build a minimal observability pipeline using OpenTelemetry spans to capture prompt, output, token count, latency, and a simple quality score. Print structured output to confirm the trace is captured correctly. Then show the same trace structure formatted for LangFuse ingestion, demonstrating the protocol difference without requiring a live account.

**Code output:** JSON-formatted trace objects printed to terminal, showing all five fields per simulated LLM call.

---

## Beat 4: Use It

**GTM redirect: Zone 1B enrichment pipeline observability.** In GTM, every Clay waterfall or AI enrichment step is an LLM call with a cost and a quality outcome. Observability answers: which enrichment prompt version produces the highest-quality ICp signals? Which account research step is burning tokens on low-value accounts? What's the per-account cost of your AI enrichment flow?

Connect the tracing mechanism to a specific enrichment workflow: tag every LLM trace with `workflow=account_research`, `prompt_version=v3`, `account_tier=enterprise`. Roll up token cost and quality scores by tier. This turns "we spent $4,000 on OpenAI last month" into "enterprise account research costs $0.12/account at 84% usable output; SMB research costs $0.03/account at 61% usable output—SMB prompt needs rework."

---

## Beat 5: Challenge

**Easy:** Add a latency threshold check to the demo pipeline. Flag any call over 5 seconds and print a warning.

**Medium:** Extend the demo to aggregate token counts by `prompt_version` across 50 simulated calls. Print a cost-per-version summary table.

**Hard:** Implement a regression detector. Store the rolling average quality score for the last 20 calls per `prompt_version`. If the next call's score drops more than 15% below the rolling average, flag it as a regression and print the alert with context (which version, what score, what the baseline was).

---

## Beat 6: Ship It

**Production checklist:**

- [ ] Every LLM call emits a trace with: prompt, output, model, token_in, token_out, latency_ms, prompt_version, workflow_tag
- [ ] Traces are queryable by workflow, prompt version, and time window
- [ ] Per-workflow cost aggregation runs daily (token count × model pricing)
- [ ] Quality eval hook fires on every output and writes a score back to the trace
- [ ] Regression alert triggers when rolling quality drops below threshold
- [ ] Retention policy is explicit (traces are large—90 days is a common default, adjust to your compliance needs)
- [ ] You have run the math on per-trace pricing at your projected monthly call volume before committing to a managed vendor

**GTM ship criteria:** Tag your first production enrichment workflow with full observability. You should be able to answer within one query: "What did the v3 account research prompt cost us this week, and what percentage of outputs scored above 0.7 quality?" If you can't answer that, the stack isn't shipped.

---

## GTM Redirect Rules

- Primary cluster: **Zone 1B — Data Foundation**, specifically AI enrichment pipeline cost and quality tracking
- Secondary cluster: **Zone 2 — Engagement**, for AI-personalized outreach output monitoring
- If the AI concept does not map to a specific GTM motion, redirect is: "foundational for any Zone using LLM calls in production—enrichment, personalization, or research"