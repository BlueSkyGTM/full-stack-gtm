# Capstone 11 — LLM Observability & Eval Dashboard

## Hook (Why This Matters)

You've built LLM-powered workflows across ten caps. Now you're flying blind the moment they hit production. Observability is the difference between "I think it works" and "here's exactly where it degrades." This capstone builds the dashboard you'll actually use to monitor every LLM call in your GTM stack.

## Concept (Mechanism)

**Observability triad:** logs (what happened), metrics (how often / how fast), traces (where in the pipeline). For LLMs specifically, the eval layer sits on top — automated heuristics that score every output on dimensions like format compliance, semantic drift, and latency percentiles. The mechanism is a structured event emitter that captures `(input, output, metadata, scores)` per call, writes to a local store, and renders aggregates to terminal.

**Sub-concepts:**
- Structured logging vs. print statements — JSON schemas let you query after the fact
- Eval heuristics: exact match, regex pattern match, JSON schema validation, semantic similarity via cosine distance
- Rolling windows for drift detection — compare last N calls to baseline distribution
- Token accounting: prompt tokens vs. completion tokens vs. cached tokens, cost attribution per GTM workflow

## Demo (Working Code)

Build a minimal observability SDK and terminal dashboard:
- `LLMObservability` class that wraps any LLM call with structured logging
- `EvalSuite` class with pluggable heuristics (format check, length bounds, semantic similarity)
- `Dashboard` class that reads the log store and prints real-time metrics table to terminal (using `rich` or plain ANSI formatting)
- Demonstrate with 20 synthetic LLM calls — show how dashboard surfaces a drift event when output quality degrades mid-stream

## Use It (GTM Application)

**GTM Cluster:** Zone 5 — AI Operations & Analytics. [CITATION NEEDED — concept: LLM observability for GTM workflow monitoring]

Every Clay waterfall, every AI email draft, every enrichment agent you've built in previous caps produces LLM calls. This observability layer answers: "Which GTM workflow is burning tokens without converting?" and "Did that prompt change last Tuesday actually improve outbound quality?" Instrument the workflows you've already shipped. The dashboard becomes your daily check on whether AI-driven GTM motions are performing within acceptable bounds.

## Ship It (Exercises)

- **Easy:** Wrap 3 existing LLM calls from previous capstones with the `LLMObservability` class. Print the structured log entries. Confirm token counts are captured.
- **Medium:** Build an eval heuristic specific to your GTM use case — e.g., "does this AI-written email contain the prospect's company name?" Run it against 50 logged outputs. Print precision/recall.
- **Hard:** Implement drift detection. Simulate a prompt regression (swap a good system prompt for a degraded one at call 30). Your dashboard must surface the drift event within 10 calls and print the alert with the suspected dimension.

## Review (Knowledge Check)

- Explain why structured logging (JSON schema) enables post-hoc queries that print statements do not.
- Compare two eval approaches: regex-based format checking vs. embedding-based semantic similarity. When does each fail?
- A GTM team reports "AI emails feel off this week." Describe the exact observability query you'd run to diagnose this — what fields, what time window, what comparison.
- Implement token cost attribution: given a log of LLM calls across 3 GTM workflows, print a cost-per-workflow breakdown table.