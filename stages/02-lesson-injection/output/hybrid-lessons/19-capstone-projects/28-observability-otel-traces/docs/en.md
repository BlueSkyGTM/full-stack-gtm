# Lesson 28: Observability with OTel GenAI Spans and Prometheus Metrics

## Learning Objectives

- Build a span data class shaped to the OpenTelemetry GenAI semantic conventions, with trace ID propagation across parent and child operations.
- Implement a JSONL exporter that writes one self-contained span record per line, roundtrip-able through `json.loads`.
- Construct counters and histograms with labeled dimensions, rendered in Prometheus text exposition format.
- Wrap any callable in a span context manager that records duration, status, exception events, and GenAI attributes.
- Diagnose an enrichment waterfall failure by correlating trace structure with metric counters across multiple vendor calls.

## The Problem

A coding agent in production produces three classes of artifact every turn: a model call, a tool execution, and a verification gate decision. None of these are useful without structured telemetry. The first failure mode is the missing trace. Something went wrong on Tuesday, but the only record is a 500-line chat log. There is no record of which tool ran, how long it took, how many tokens went into the prompt, or whether the gate refused anything. The agent author has to guess — and guessing at production behavior is how outages extend from minutes to hours.

The second failure mode is the unparseable trace. The harness wrote spans but used its own ad-hoc field names — `model_name` instead of `gen_ai.request.model`, `tokens_in` instead of `gen_ai.usage.input_tokens`. Nothing in Grafana, Honeycomb, Jaeger, or the local CLI can read them without a custom parser that somebody has to write and maintain. Whatever tooling already exists in the team's stack is wasted because the spans are non-standard. The OpenTelemetry GenAI semantic conventions exist precisely to solve this: they define a fixed vocabulary of attribute names so that any backend can parse LLM traces without custom integration work.

The third failure mode is the unaggregated metric. You can see one slow tool call in a trace, but you cannot answer "what is the p99 latency for GPT-4 calls over the last hour?" without computing across all traces. That query should be a single PromQL expression against a pre-computed histogram — not a scan of a JSONL file. Counters track totals (requests, tokens, errors), histograms track distributions (latency, token counts), and gauges track point-in-time state (active requests, queue depth). Without these three primitives exposed on an endpoint, you have no dashboards, no alerts, and no capacity planning.

The GTM parallel is direct. When a Clay enrichment waterfall runs Clearbit, Hunter, and Apollo in sequence for a list of 10,000 companies, each vendor call is a span in a trace. If Hunter's API degrades from 200ms to 3s, you need the span's `http.response.status_code` and duration to surface it in a dashboard — not a Slack message from the SDR team twelve hours later saying "enrichment seems slow today." [CITATION NEEDED — concept: Clay enrichment waterfall latency observability]

## The Concept

A span represents a single operation — an LLM inference, a tool call, a retrieval step, an HTTP request to an enrichment vendor. Each span carries a trace ID that links it to every other span in the same logical request, a span ID that uniquely identifies it within that trace, a parent span ID that establishes the causal tree, a span kind (client, server, or internal), and a bag of key-value attributes. The OpenTelemetry GenAI semantic conventions standardize the attribute names for AI operations: `gen_ai.request.model` for the model name, `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` for token counts, `gen_ai.response.finish_reason` for why generation stopped, `gen_ai.operation.name` for the operation type (chat, completion, embedding). Any backend that understands these conventions — Jaeger, Honeycomb, Datadog, Grafana Tempo — can parse your LLM traces without custom configuration.

Prometheus pulls numeric time-series from an `/metrics` endpoint your process exposes. Three primitives cover the majority of observability needs. A **counter** is monotonically increasing — total requests, total tokens consumed, total errors. It never goes down. A **histogram** records the distribution of a value across configurable buckets — latency in ranges like `[0.1s, 0.25s, 0.5s, 1.0s, 2.5s, 5.0s]`, token count distributions. A **gauge** is a point-in-time value that can go up or down — active in-flight requests, queue depth, rate limit remaining. Each metric carries labels (model name, endpoint, status code, vendor name) so you can slice and filter by dimension in PromQL queries.

The wiring pattern is: instrument once, export to two sinks. The tracer emits spans to a JSONL file (or an OTLP collector, or a tracing backend). The same instrumentation points also update Prometheus metrics — the span's attributes become counter labels and histogram observations. This avoids double instrumentation: the same `gen_ai.request.model` attribute that lands in the span also becomes the `model` label on the Prometheus counter. The trace gives you the per-request story; the metrics give you the aggregate story.

```mermaid
flowchart LR
    A[LLM Call] --> B[Span Context Manager]
    B --> C{GenAI Attributes}
    C -->|gen_ai.request.model| D[JSONL Exporter]
    C -->|gen_ai.usage.*| D
    C -->|gen_ai.response.*| D
    B --> E[Metrics Registry]
    E -->|counter.inc| F[requests_total]
    E -->|counter.inc| G[tokens_total]
    E -->|histogram.observe| H[latency_seconds]
    D --> I[traces.jsonl]
    F --> J[/metrics endpoint]
    G --> J
    H --> J
    J --> K[Prometheus Scrape]
```

Why OpenTelemetry and not raw logging? Correlation. A trace ID on a span links the LLM call in your enrichment pipeline back to the webhook that triggered it. Without that linkage, you have fragmented logs scattered across services with no way to reconstruct the causal chain. With it, you have a tree of operations — webhook received, data parsed, vendor one called, vendor two called, result written — each node carrying its own duration, status, and attributes. The trace ID is the thread that stitches them together.

## Build It

The implementation has four components: a `Span` dataclass with GenAI attributes, a `JSONLExporter` that writes one span per line, `Counter` and `Histogram` classes that render Prometheus text format, and a `Tracer` that provides a context manager wrapping any callable. The code uses only Python stdlib — `json`, `time`, `uuid`, `threading`, `dataclasses`, `contextlib` — and runs offline with no external collector.

```python
import json
import time
import uuid
import threading
import os
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Optional

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    start_time_unix_nano: int
    end_time_unix_nano: Optional[int] = None
    status: str = "UNSET"
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @property
    def duration_ms(self):
        if self.end_time_unix_nano is None:
            return None
        return (self.end_time_unix_nano - self.start_time_unix_nano) / 1e6


class JSONLExporter:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()

    def export(self, span):
        record = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "kind": span.kind,
            "start_time_unix_nano": span.start_time_unix_nano,
            "end_time_unix_nano": span.end_time_unix_nano,
            "status": span.status,
            "attributes": span.attributes,
            "events": span.events,
        }
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record


class Counter:
    def __init__(self, name, help_text, label_names=None):
        self.name = name
        self.help_text = help_text
        self.label_names = label_names or []
        self._values = {}
        self._lock = threading.Lock()

    def inc(self, value=1, **labels):
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def render(self):
        lines = [f"# HELP {self.name} {self.help_text}",
                 f"# TYPE {self.name} counter"]
        for key, val in sorted(self._values.items()):
            if self.label_names:
                pairs = ",".join(f'{n}="{v}"' for n, v in zip(self.label_names, key))
                lines.append(f'{self.name}{{{pairs}}} {val}')
            else:
                lines.append(f'{self.name} {val}')
        return "\n".join(lines)


class Histogram:
    def __init__(self, name, help_text, buckets, label_names=None):
        self.name = name
        self.help_text = help_text
        self.buckets = sorted(buckets)
        self.label_names = label_names or []
        self._observations = {}
        self._lock = threading.Lock()

    def observe(self, value, **labels):
        key = tuple(labels.get(l, "") for l in self.label_names)
        with self._lock:
            if key not in self._observations:
                self._observations[key] = {
                    "count": 0,
                    "sum": 0.0,
                    "buckets": {b: 0 for b in self.buckets},
                }
            obs = self._observations[key]
            obs["count"] += 1
            obs["sum"] += value
            for b in self.buckets:
                if value <= b:
                    obs["buckets"][b] += 1

    def render(self):
        lines = [f"# HELP {self.name} {self.help_text}",
                 f"# TYPE {self.name} histogram"]
        for key, obs in sorted(self._observations.items()):
            if self.label_names:
                base = ",".join(f'{n}="{v}"' for n, v in zip(self.label_names, key))
                bucket_prefix = "{" + base + ","
                full_labels = "{" + base + "}"
            else:
                bucket_prefix = "{"
                full_labels = ""
            for b in self.buckets:
                lines.append(f'{self.name}_bucket{bucket_prefix}le="{b}"}} {obs["buckets"][b]}')
            lines.append(f'{self.name}_bucket{bucket_prefix}le="+Inf"}} {obs["count"]}')
            lines.append(f'{self.name}_count{full_labels} {obs["count"]}')
            lines.append(f'{self.name}_sum{full_labels} {obs["sum"]}')
        return "\n".join(lines)


class MetricsRegistry:
    def __init__(self):
        self._metrics = {}

    def register(self, metric):
        self._metrics[metric.name] = metric
        return metric

    def render(self):
        return "\n\n".join(m.render() for m in self._metrics.values())


class Tracer:
    def __init__(self, exporter, service_name="default"):
        self.exporter = exporter
        self.service_name = service_name
        self._current = threading.local()

    def _new_trace_id(self):
        return uuid.uuid4().hex

    def _new_span_id(self):
        return uuid.uuid4().hex[:16]

    @property
    def current_span(self):
        return getattr(self._current, "span", None)

    @contextmanager
    def start_span(self, name, kind="internal", attributes=None):
        parent = self.current_span
        span = Span(
            trace_id=parent.trace_id if parent else self._new_trace_id(),
            span_id=self._new_span_id(),
            parent_span_id