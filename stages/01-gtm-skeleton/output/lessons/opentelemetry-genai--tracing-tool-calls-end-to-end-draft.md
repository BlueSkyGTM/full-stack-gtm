# OpenTelemetry GenAI — Tracing Tool Calls End-to-End

---

## Beat 1: Hook

When an agent calls three tools in sequence and returns a wrong answer, you need to know *which call* failed and *what it passed*. OpenTelemetry's GenAI semantic conventions give you a standard schema for capturing that chain as a single trace — model invocation, tool selection, tool execution, and response — without custom logging hacks.

---

## Beat 2: Concept

Introduce the three primitives: **traces**, **spans**, and **span attributes**. Explain how OpenTelemetry's GenAI semantic conventions (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.response.finish_reason`, etc.) map each step of a tool-calling loop to a named, timestamped span. Contrast this with ad-hoc `print()` debugging: traces survive across process boundaries, carry structured metadata, and aggregate in a backend (Jaeger, Honeycomb, Grafana Tempo) for filtering.

---

## Beat 3: Mechanism

Walk through the span lifecycle for a single tool-calling turn:

1. **LLM span** wraps the model request/response. Attributes capture model name, token counts, finish reason.
2. **Tool call span** is a child of the LLM span. Attributes capture tool name, arguments (scrubbed of PII), and return value.
3. **Aggregation**: if the model calls tools iteratively, each iteration creates a new LLM span under a parent **agent span**, producing a nested tree.

Explain how the OpenTelemetry Python SDK (`opentelemetry-sdk`, `opentelemetry-exporter-otlp`) bootstraps a tracer provider, and how the GenAI conventions define which attributes land on which span type. Note: the GenAI semantic conventions are still in semconv `1.28+` and subject to change — flag this explicitly.

---

## Beat 4: Use It

**GTM Redirect**: This maps to **Zone 3 — Pipeline Operations**. When a GTM team runs a multi-step enrichment waterfall (e.g., Clay → LLM classify → web scrape → CRM write), each tool call is a span. Observability surfaces which step is slow, which returns errors, and where tokens are burned.

**Exercise hooks**:
- *Easy*: Instrument a single LLM call with an OTel span, export to console, and print the trace ID.
- *Medium*: Wrap a two-tool agent loop (tool A feeds tool B) with nested spans; verify the parent-child relationship in the exported JSON.
- *Hard*: Add a span processor that redacts PII from tool-call arguments before export; demonstrate with a realistic enrichment payload containing an email address.

---

## Beat 5: Ship It

**GTM Redirect**: Same Zone 3 cluster — production agent monitoring. Cover sampling strategies (head-based vs. tail-based), exporter configuration (OTLP/gRPC vs. OTLP/HTTP), and cardinality hygiene (avoid unbounded attribute values like raw user input as span attributes). Address the operational question: "What do I alert on?" — suggest `gen_ai.response.finish_reason == "error"` and tool-call latency percentiles as starting signals.

**Exercise hooks**:
- *Easy*: Configure a `BatchSpanProcessor` with a rate-limiting sampler; confirm sampling behavior by generating 100 traces and counting exported spans.
- *Medium*: Export traces to a local Jaeger container via OTLP; query the Jaeger API for traces where a specific tool returned an error.
- *Hard*: Implement a tail-based sampler that always retains traces containing tool-call spans with duration > 2s; validate with synthetic slow tool calls.

---

## Beat 6: Stretch

Introduce **baggage propagation**: passing correlated context (e.g., `lead_id`, `campaign_id`) across agent boundaries so that a single trace spans from initial lead ingestion through every tool call to final CRM write. Note the security implication — baggage is propagated in headers and visible to downstream services; do not put secrets in baggage. [CITATION NEEDED — concept: OpenTelemetry baggage security best practices for multi-tenant GTM pipelines]

**Exercise hooks**:
- *Medium*: Attach `campaign_id` as baggage to an agent trace; verify it appears as an attribute on every child span created downstream.
- *Hard*: Build a minimal two-service setup (service A calls service B via HTTP) where the trace and baggage propagate across the boundary; confirm a single trace ID appears in both services' exported spans.