# Lesson 28: Observability with OTel GenAI Spans and Prometheus Metrics

## Hook

When an LLM call silently degrades latency from 800ms to 6s, or token costs spike without a code change, you need structured telemetry — not log grep. This lesson wires OpenTelemetry's GenAI semantic conventions and Prometheus counters into a working pipeline so every inference is traceable, measurable, and alertable.

## Concept

**Span model:** A span represents a single operation — an LLM inference, a tool call, a retrieval step. Each span carries a trace ID (links operations across services), a span kind (client/server/internal), and a bag of key-value attributes. OTel's GenAI semantic conventions standardize attribute names: `gen_ai.request.model`, `gen_ai.response.finish_reason`, `gen_ai.usage.input_tokens`. This means any backend — Jaeger, Honeycomb, Datadog — can parse your LLM traces without custom parsing.

**Metrics model:** Prometheus pulls numeric time-series from an `/metrics` endpoint your process exposes. Three primitives matter here: **counters** (monotonically increasing — total tokens consumed, total requests), **histograms** (distribution of values — latency buckets, token count distributions), **gauges** (point-in-time values — active requests, queue depth). Each metric carries labels (model name, endpoint, status code) so you can slice by dimension.

**The wiring pattern:** Instrument once → export to two sinks. The OTel SDK emits spans to a collector (or directly to a tracing backend). The same SDK or a companion Prometheus client exposes `/metrics`. The span's attributes become Prometheus labels when you bridge the two. This avoids double instrumentation.

**Why OTel and not raw logging:** Correlation. A trace ID on a span links the LLM call in your enrichment pipeline back to the Clay webhook that triggered it. Without that linkage, you have fragmented logs. With it, you have a因果链.

## Demo

Build a minimal pipeline that:
1. Wraps an LLM call in an OTel span with GenAI attributes (model, tokens, finish reason)
2. Increments a Prometheus counter for total requests and records latency in a histogram
3. Exposes `/metrics` on `localhost:8000`
4. Prints the trace ID and span ID to stdout to confirm instrumentation fired
5. Prints a curl command the practitioner can run to see Prometheus metrics

The demo uses `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, and `prometheus-client`. All local — no external collector required. Observable output is the printed trace/span IDs and the raw `/metrics` text.

## Use It

**GTM cluster: Zone 3 — Pipeline Health and Enrichment Reliability**

When a Clay waterfall runs enrichment through three vendors (Clearbit → Hunter → Apollo), each HTTP call is a span. The trace ID ties all three to the same enrichment request. If Hunter's latency spikes, the span's `http.response.status_code` and duration surface it in your dashboard — not in a Slack thread 12 hours later.

Prometheus counters track: enrichment success rate by vendor, tokens consumed per ICP classification, and error budget burn rate. If your ICP classifier's token cost per classification doubles overnight, the `gen_ai_usage_input_tokens` counter segmented by model catches it before the billing invoice does.

**Concrete wiring for GTM practitioners:**
- Label every Prometheus metric with `pipeline=clay_waterfall`, `step=clearbit`, `record_type=company`
- Set span attributes `gtm.enrichment.source`, `gtm.enrichment.field` (e.g., `source=clearbit`, `field=employee_count`)
- Alert on: p99 latency > 3s on any enrichment step, error rate > 5% on any vendor, token cost anomaly > 2σ from 7-day rolling mean

[CITATION NEEDED — concept: OTel semantic conventions for enrichment pipeline labeling standard]

## Ship It

**Exporter configuration:** In production, spans export to an OTel Collector (sidecar or gateway), which forwards to your tracing backend. The Collector handles batching, retry, and tail-based sampling (keep 100% of error spans, sample 10% of successful ones). Prometheus scrapes `/metrics` on a 15s interval — no push required.

**Cardinality warning:** Every unique label combination is a new time series. If you label by `company_domain`, you get one series per company. This detonates your Prometheus storage. Label by low-cardinality dimensions only: `model`, `vendor`, `status_code`, `pipeline_step`. High-cardinality data (domain, user ID) goes in span attributes, not metric labels.

**Sampling strategy:** Head-based sampling (decide at span creation) is simpler but blind to downstream errors. Tail-based sampling (decide after the trace completes) requires a Collector but lets you keep all error traces and drop successful ones. For GTM pipelines where errors are rare but critical, tail-based pays for itself.

**Dashboard layout:** Three panels minimum — latency histogram (p50/p95/p99), token cost counter (cumulative and rate), error counter by vendor. Alert on error budget burn rate, not absolute thresholds.

## Tighten It

**Easy:** Add a new Prometheus histogram for `enrichment_duration_seconds` with labels `vendor` and `status`. Verify it appears at `/metrics`.

**Medium:** Implement tail-based sampling in an OTel Collector config that keeps 100% of spans where `status.code == ERROR` and samples 10% of `OK` spans. Confirm the ratio in your tracing backend.

**Hard:** Build a bridge that extracts `gen_ai.usage.total_tokens` from completed spans and emits it as a Prometheus gauge labeled by model. Validate that the gauge value matches the sum of input + output tokens across a 5-minute window. Output the discrepancy if any.

---

**Learning Objectives (testable):**

1. Configure an OTel span with GenAI semantic conventions (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.finish_reason`) and verify the attributes appear in exported trace data.
2. Instrument a Python function with Prometheus counters and histograms, expose `/metrics`, and confirm scrape output contains the expected metric names and label dimensions.
3. Diagnose a latency regression by querying span duration attributes filtered by `gen_ai.request.model` across a trace.
4. Evaluate metric cardinality for a given label set and eliminate any label that produces unbounded unique values.
5. Implement a sampling rule that preserves 100% of error spans and reduces successful span volume by a configurable factor.