# OpenTelemetry GenAI Semantic Conventions

## Hook

You're paying for tokens but can't answer "which model costs us the most per qualified lead?" OpenTelemetry's GenAI semantic conventions give you a standard schema for attributing LLM cost, latency, and quality back to the pipeline that invoked them.

## Concept

Semantic conventions are a pre-agreed dictionary of span attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.) that every OTel-instrumented SDK emits identically regardless of provider. The mechanism: you instrument once using the conventions, and any OTel-compatible backend (Jaeger, Honeycomb, Grafana Tempo) can parse, aggregate, and alert on those attributes without custom parsing. Covers chat completions, embeddings, and tool/function calls.

## Demonstration

Instrument a minimal Anthropic call using the OpenTelemetry GenAI conventions via the Python `opentelemetry-instrumentation-anthropic` package (or manual span creation if the auto-instrumentation package is not yet released). Print the emitted span attributes to stdout to confirm the convention keys appear correctly. Show a second call to an OpenAI-compatible endpoint to prove the same attribute names fire regardless of provider.

## Use It

Map `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` grouped by `gen_ai.request.model` back to your enrichment pipeline runs. This is the observability layer for the **AI Enrichment** cluster in Zone 2 — when your Clay waterfall or custom enrichment agent calls an LLM to score leads, these conventions let you tag cost to source, not just to model. Exercise hook: write a simple attribute grouping script that reads exported OTel JSON and prints cost-per-enrichment-run (easy), attribute a multi-step enrichment pipeline's total token spend to the original input company (medium), correlate token spend with enrichment outcome quality to identify the cost-efficiency breakpoint per model (hard).

## Ship It

Export GenAI spans via OTLP to your observability backend. Set a dashboard filter on `gen_ai.system` and `gen_ai.request.model`, then create an alert on `gen_ai.usage.total_tokens` exceeding a threshold per enrichment batch. Exercise hook: configure an OTLP file exporter and write a script that parses the output file to report total tokens per model (easy), wire an OTLP HTTP exporter to a local Jaeger instance and verify traces appear with correct GenAI attributes (medium), implement a token-budget guard that reads accumulated `gen_ai.usage.total_tokens` from the last hour and blocks further LLM calls when a threshold is exceeded (hard).

## Evaluate

Quiz hooks: identify which span attributes belong to the GenAI conventions vs. generic OTel HTTP spans (easy), predict the attribute names emitted for a tool-calling sequence given the conventions spec (medium), design an attribute-extension strategy for a multi-agent pipeline where sub-agents call different models but share a correlation ID — explain what you'd extend vs. what you'd reuse from the existing conventions (hard).