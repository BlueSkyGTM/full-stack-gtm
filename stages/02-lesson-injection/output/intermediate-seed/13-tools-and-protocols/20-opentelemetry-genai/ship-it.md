## Ship It

Moving from local console output to a production backend changes the operational surface area. You need a sampling strategy (you cannot export every span at high volume), an exporter configuration (OTLP/gRPC vs OTLP/HTTP), and cardinality discipline (unbounded attribute values will tank your backend's query performance). These are Zone 3 concerns — the same zone as deployment and CI/CD infrastructure for GTM pipelines [CITATION NEEDED — concept: Zone 13 Deployment Infrastructure mapping to observability].

**Sampling**: head-based sampling decides at trace start whether to export, using a fixed ratio (e.g., 10% of traces). It is cheap and deterministic but blind to outcomes — it will drop the interesting error traces at the same rate as boring successes. Tail-based sampling inspects the completed trace before deciding, so you can keep 100% of traces with `gen_ai.response.finish_reason == "error"` and sample the rest at 5%. The tradeoff is memory: the collector must buffer the full trace before deciding. For GTM enrichment pipelines where error rates are low but each error is expensive to debug, tail-based sampling with an error-first rule is the right default.

**Exporter**: OTLP/gRPC is the default for most backends and handles high throughput well. OTLP/HTTP is simpler to firewall (single port, standard HTTP) and works better behind corporate proxies that block gRPC. Both serialize the same protobuf payloads — the choice is transport, not data model.

**Cardinality hygiene**: this is where production tracing silently fails. If you set `gen_ai.tool.arguments` to the raw user input on every span, and user input contains free-text queries, you create a unique attribute value per request. Your backend builds an index entry for each, and memory grows unbounded. The fix: hash, truncate, or bucket. Store the first 50 characters, or hash to a fixed-length identifier, or extract a category label instead of raw text.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
import time
import random

resource = Resource.create({
    "service.name": "gtm-enrichment-agent",
    "service.version": "1.4.0",
    "deployment.environment": "production",
})

sampler = TraceIdRatioBased(0.1)

provider = TracerProvider(resource=resource, sampler=sampler)
batch_processor = BatchSpanProcessor(
    ConsoleSpanExporter(),
    max_queue_size=2048,
    max_export_batch_size=512,
    export_timeout_millis=30000,
)
provider.add_span_processor(batch_processor)

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("gtm.agent.production")

def truncate_for_cardinality(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...[truncated]"

ERROR_MESSAGES = [
    "Tool execution failed: connection timeout to MCP server",
    "Tool execution failed: rate limit exceeded (429)",
    "Tool execution succeeded",
]

for i in range(10):
    with tracer.start_as_current_span("agent.invoke_agent") as agent_span:
        agent_span.set_attribute("gen_ai.agent.name", "enrichment-agent")
        agent_span.set_attribute("lead.batch_id", f"batch_{i // 3}")

        with tracer.start_as_current_span("llm.chat", kind=trace.SpanKind.CLIENT) as llm_span:
            llm_span.set_attribute("gen_ai.system", "openai")
            llm_span.set_attribute("gen_ai.request.model", "gpt-4o")
            llm_span.set_attribute("gen_ai.usage.input_tokens", random.randint(50, 200))
            llm_span.set_attribute("gen_ai.usage.output_tokens", random.randint(20, 100))

            if random.random() < 0.2:
                llm_span.set_attribute("gen_ai.response.finish_reason", "length")
                llm_span.set_status(Status(StatusCode.ERROR, "Response truncated"))
            else:
                llm_span.set_attribute("gen_ai.response.finish_reason", "stop")

        with tracer.start_as_current_span("tool.execute") as tool_span:
            raw_input = f"Enrich lead from source: campaign_{random.randint(1000, 9999)} extra context: " + "x" * 200
            tool_span.set_attribute("gen_ai.tool.name", "crm_write")
            tool_span.set_attribute("gen_ai.tool.arguments", truncate_for_cardinality(raw_input))
            tool_span.set_attribute("gen_ai.tool.result", random.choice(ERROR_MESSAGES))

        time.sleep(0.05)

print("\n--- Production trace batch exported ---")
print(f"Sampler ratio: 0.1 (roughly 1 of 10 traces exported)")
print(f"Batch processor: queues spans, exports in batches of 512")
print("Cardinality: arguments truncated to 80 chars, batch_id bucketed")
```

Run this and observe: roughly one in ten traces appears in the console output (the sampler drops the rest). The `lead.batch_id` attribute is bucketed to three values — `batch_0`, `batch_1`, `batch_2` — not ten unique values. The tool arguments are truncated, preventing unbounded cardinality from the variable-length input string. This is the shape of a production configuration.

**What to alert on**: start with two signals. First, `gen_ai.response.finish_reason` in `["length", "error", "content_filter"]` — these indicate the model did not complete normally, and in a GTM pipeline they usually mean the enrichment step produced garbage downstream. Second, tool-call latency at the 95th percentile — if `tool.execute` spans for `crm_write` suddenly jump from 200ms to 5s, your CRM API is degraded or rate-limited. These two alerts catch the majority of agent failures in practice. Add token-cost alerts (`gen_ai.usage.input_tokens + gen_ai.usage.output_tokens` summed per trace) once you have baseline data.