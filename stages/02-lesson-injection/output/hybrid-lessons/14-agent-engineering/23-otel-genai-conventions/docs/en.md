# OpenTelemetry GenAI Semantic Conventions

## Learning Objectives

- Implement GenAI semantic conventions on a raw LLM call using manual OTel spans with the correct attribute names for provider, model, and token usage
- Compare span attributes emitted by different providers under the same convention schema to prove cross-provider aggregation works
- Configure content-capture opt-in via `OTEL_SEMCONV_STABILITY_OPT_IN` and explain why prompt body capture defaults to off
- Compute cost-per-enrichment-run by grouping GenAI span attributes by model and pipeline correlation ID
- Trace a multi-step enrichment pipeline's token spend back to the originating input record using parent-child span relationships

## The Problem

Every LLM provider SDK instruments telemetry differently. Anthropic's SDK logs token counts under one key shape, OpenAI's under another, and Bedrock wraps everything behind an AWS-specific event format. If you run a Clay waterfall enrichment that calls three different models in sequence — say, Claude to classify industry, GPT-4o to score intent, then Claude again to draft a personalized opener — you end up with three incompatible telemetry streams. Your observability backend shows three separate traces with no shared vocabulary. You cannot answer "how much did enriching Acme Corp cost across all model calls?" without writing custom parsers per provider.

The practical consequence is that cost attribution becomes a manual exercise. You export billing CSVs from each provider console, manually reconcile timestamps against pipeline runs, and still miss the per-call granularity that tells you *which* step in the enrichment burned the most tokens. This is the observability gap the OpenTelemetry GenAI Special Interest Group set out to close in April 2024.

The deeper problem is not just parsing — it is semantic alignment. Even if every provider exported JSON, "input tokens" in Anthropic's API and "prompt tokens" in OpenAI's API refer to the same concept but use different names. Without a shared dictionary, every dashboard, alert rule, and cost report is bespoke. The GenAI semantic conventions define that dictionary so that a span emitted by an Anthropic call and a span emitted by an OpenAI call carry identically named attributes for the same underlying concepts.

## The Concept

Semantic conventions are a pre-agreed dictionary of span attribute names and values. The OpenTelemetry GenAI SIG defines attributes like `gen_ai.system` (the provider: `anthropic`, `openai`, `aws.bedrock`), `gen_ai.request.model` (the model ID string), `gen_ai.usage.input_tokens`, and `gen_ai.usage.output_tokens`. Any OTel-instrumented SDK that adopts these conventions emits the same keys for the same concepts. Your backend — whether Jaeger, Honeycomb, Datadog, or Grafana Tempo — parses, aggregates, and alerts on those attributes without provider-specific custom logic.

The conventions define three span categories. **Model/client spans** wrap raw LLM API calls and carry the usage attributes you need for cost analysis. **Agent spans** (`invoke_agent`) wrap the orchestration loop of an agent framework — LangGraph, CrewAI, a custom ReAct loop — and carry the agent's name and configuration. **Tool spans** wrap individual tool executions invoked by the agent. These nest as parent-child: an agent span is the parent of the model and tool spans it orchestrates, which means you can walk the span tree to attribute total cost back to the agent run that incurred it.

Agent span naming follows a template: `invoke_agent {gen_ai.agent.name}` if the agent has a name, otherwise just `invoke_agent`. The span kind distinguishes how the agent runs. A **CLIENT** span means the agent is a remote service — OpenAI Assistants API, Bedrock Agents — where your code calls an external endpoint. An **INTERNAL** span means the agent runs in-process — LangChain, CrewAI, or a hand-rolled loop. This distinction matters for sampling: remote agent calls cross network boundaries and may need different sampling rates than in-process orchestration.

Content capture — logging the actual prompt and response text — is opt-in by default. The conventions specify that sensitive content (user prompts, model completions) should not be captured unless you explicitly enable it. You opt in by setting the environment variable `OTEL_SEMCONV_STABILITY_OPT_IN` to include `gen_ai` and configuring capture settings in your instrumentation library. The recommendation is to capture external references (document IDs, image URLs) rather than raw content, reducing both privacy risk and storage cost. This is particularly relevant for GTM pipelines where prompts may contain customer PII scraped from LinkedIn or company websites.

The span tree below shows how GenAI attributes attach to different span types in a typical enrichment pipeline:

```mermaid
flowchart TD
    ROOT["pipeline.run enrichment_run_42<br/>(custom root span)"] --> AGENT["invoke_agent lead_scorer<br/>(INTERNAL)"]
    AGENT --> CHAT1["chat<br/>(model/client span)"]
    AGENT --> TOOL["tool_call enrich_company<br/>(tool span)"]
    AGENT --> CHAT2["chat<br/>(model/client span)"]
    CHAT1 --> A1["gen_ai.system = anthropic<br/>gen_ai.request.model = claude-3-5-sonnet<br/>gen_ai.usage.input_tokens = 1240<br/>gen_ai.usage.output_tokens = 340"]
    CHAT2 --> A2["gen_ai.system = openai<br/>gen_ai.request.model = gpt-4o<br/>gen_ai.usage.input_tokens = 890<br/>gen_ai.usage.output_tokens = 210"]
    TOOL --> A3["tool.name = enrich_company<br/>tool.input.domain = acme.com"]
```

Both model/client spans emit the same attribute names (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) despite calling different providers. This is the mechanism that makes cross-provider aggregation possible — your cost dashboard groups by `gen_ai.request.model` and sums `gen_ai.usage.input_tokens` regardless of which provider SDK emitted the span.

## Build It

To see the conventions in action, you need the OpenTelemetry SDK and a console exporter that prints span attributes to stdout. Install the required packages first:

```bash
pip install opentelemetry-api opentelemetry-sdk
```

The code below creates two manual spans that simulate LLM calls to different providers. Each span carries the GenAI convention attributes. The `SimpleSpanProcessor` with `ConsoleSpanExporter` prints each span's full attribute set as JSON when it ends, so you can verify the convention keys appear correctly.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
exporter = ConsoleSpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("genai-conventions-demo")


def simulate_llm_call(provider_name, model, input_text, input_tokens, output_tokens):
    with tracer.start_as_current_span("chat") as span:
        span.set_attribute("gen_ai.system", provider_name)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        response = f"[{model}] Processed {len(input_text)} chars of input"
        span.set_attribute("gen_ai.response.finish_reason", "stop")
        return response


result_anthropic = simulate_llm_call(
    "anthropic",
    "claude-3-5-sonnet",
    "Score this lead: Acme Corp, 500 employees, Series B",
    input_tokens=1240,
    output_tokens=340,
)

result_openai = simulate_llm_call(
    "openai",
    "gpt-4o",
    "Draft a personalized opener for Acme Corp based on their recent funding",
    input_tokens=890,
    output_tokens=210,
)

print(f"\nAnthropic result: {result_anthropic}")
print(f"OpenAI result: {result_openai}")
print(f"\nBoth spans emitted identical attribute names despite different providers.")
```

When you run this, the console exporter prints two JSON span objects. Look for the `attributes` key in each — both spans contain `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, and `gen_ai.usage.output_tokens`. The values differ (different provider names, different model IDs, different token counts) but the keys are identical. That is the entire point of semantic conventions: same keys, different values, uniform parsing.

Now let us build a realistic enrichment pipeline span tree. This creates a parent pipeline span, a child agent span, and two grandchild model spans — the structure you would see in a Clay waterfall enrichment that calls multiple models per company:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("enrichment-pipeline")


def enrich_company(domain, run_id):
    with tracer.start_as_current_span("pipeline.run") as pipeline:
        pipeline.set_attribute("pipeline.run_id", run_id)
        pipeline.set_attribute("pipeline.input.domain", domain)
        pipeline.set_attribute("pipeline.step", "enrich_company")

        with tracer.start_as_current_span(
            "invoke_agent lead_scorer", kind=SpanKind.INTERNAL
        ) as agent:
            agent.set_attribute("gen_ai.agent.name", "lead_scorer")

            with tracer.start_as_current_span("chat") as step1:
                step1.set_attribute("gen_ai.system", "anthropic")
                step1.set_attribute("gen_ai.request.model", "claude-3-5-sonnet")
                step1.set_attribute("gen_ai.operation.name", "chat")
                step1.set_attribute("gen_ai.usage.input_tokens", 1240)
                step1.set_attribute("gen_ai.usage.output_tokens", 340)

            with tracer.start_as_current_span("chat") as step2:
                step2.set_attribute("gen_ai.system", "openai")
                step2.set_attribute("gen_ai.request.model", "gpt-4o")
                step2.set_attribute("gen_ai.operation.name", "chat")
                step2.set_attribute("gen_ai.usage.input_tokens", 890)
                step2.set_attribute("gen_ai.usage.output_tokens", 210)

    return f"Enriched {domain} (run {run_id}): 2 model calls, 2680 total tokens"


result = enrich_company("acme.com", "enrich_2024_001")
print(f"\n{result}")
```

The output shows four spans with a clear parent-child hierarchy. The pipeline span is the root. The agent span is its child. Both model spans are children of the agent span. This tree structure is what lets you walk from any leaf span (a single LLM call) up to its pipeline run — and that is how you attribute token cost back to the input company that triggered the enrichment.

## Use It

The GenAI semantic conventions map directly onto the **AI Enrichment** cluster in Zone 2. When your Clay waterfall or custom enrichment agent calls an LLM to score leads, extract firmographics, or draft personalization, each call emits a model/client span with `gen_ai.request.model` and token usage attributes. By adding a custom `pipeline.run_id` attribute to every span in the tree, you create a join key that links every token spent back to the specific enrichment run — and therefore to the specific input company.

This is the observability layer for Zone 14's cost management mandate: "Every Clay credit is a token cost — optimize like you would LLM calls" [CITATION NEEDED — concept: Clay credit-to-token cost equivalence]. Without semantic conventions, you cannot build a cost dashboard that works across models. With them, the same SpanQL or TraceQL query aggregates spend across Anthropic, OpenAI, and any future provider your pipeline adopts — because they all speak the same attribute vocabulary.

The runnable slice below demonstrates the mechanism: manual OTel spans with GenAI convention attributes attached to a enrichment pipeline that calls two providers. The `pipeline.run_id` acts as the correlation key. Multiply token counts by per-model rates and you get per-run cost without touching a provider billing console.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("gtm-cost-tracker")

RATES = {
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "gpt-4o": {"input": 0.005, "output": 0.015},
}

def enrichment_run(run_id, domain, model_calls):
    with tracer.start_as_current_span("pipeline.run") as root:
        root.set_attribute("pipeline.run_id", run_id)
        root.set_attribute("pipeline.input.domain", domain)
        total = 0.0
        for mc in model_calls:
            with tracer.start_as_current_span("chat") as span:
                span.set_attribute("gen_ai.system", mc["system"])
                span.set_attribute("gen_ai.request.model", mc["model"])
                span.set_attribute("gen_ai.usage.input_tokens", mc["in"])
                span.set_attribute("gen_ai.usage.output_tokens", mc["out"])
                r = RATES[mc["model"]]
                total += (mc["in"] * r["input"] + mc["out"] * r["output"]) / 1000
        root.set_attribute("pipeline.total_cost_usd", round(total, 4))
        return total

cost = enrichment_run("enrich_042", "acme.com", [
    {"system": "anthropic", "model": "claude-3-5-sonnet", "in": 1240, "out": 340},
    {"system": "openai", "model": "gpt-4o", "in": 890, "out": 210},
])
print(f"acme.com enrichment cost: ${cost:.4f}")
```

The console exporter prints three spans. The root `pipeline.run` span carries `pipeline.total_cost_usd`, `pipeline.run_id`, and `pipeline.input.domain`. The two child `chat` spans carry the GenAI convention attributes. Any backend that ingests these spans — Datadog, Honeycomb, Tempo — can now group by `gen_ai.request.model` to show cost-per-model, or group by `pipeline.run_id` to show cost-per-enrichment-run. That query works regardless of which provider SDK emitted the span, because the attribute names are identical.

## Exercises

**Exercise 1 — Add a Third Provider (Medium).** Extend the `enrichment_run` function to include a third model call to `aws.bedrock` using model `anthropic.claude-3-haiku`. Add the rate entry to the `RATES` dictionary (input: 0.0008, output: 0.0016 per 1K tokens). Run the pipeline and confirm the third `chat` span appears with `gen_ai.system = aws.bedrock` and that the total cost includes all three calls. Then write a one-line Python comprehension that sums `gen_ai.usage.input_tokens` across all three child spans to verify cross-provider aggregation yields a single number.

**Exercise 2 — Per-Step Cost Attribution (Hard).** Modify the enrichment pipeline so each `chat` span includes a `pipeline.step_name` attribute (e.g., `"industry_classify"`, `"intent_score"`, `"opener_draft"`). Export the spans to a JSON file. Then write a separate script that reads the JSON, groups spans by `pipeline.step_name`, and prints a table showing step name, model used, input tokens, output tokens, and cost per step. The goal is to identify which enrichment step is the most expensive — the exact question a GTM engineer needs to answer when deciding whether to downgrade a step to a cheaper model.

## Key Terms

- **Semantic Conventions** — A pre-agreed dictionary of OTel span attribute names and value formats that ensures different instrumentation libraries emit compatible telemetry for the same concepts.
- **`gen_ai.system`** — The convention attribute identifying the LLM provider (e.g., `anthropic`, `openai`, `aws.bedrock`). Enables provider-level grouping in dashboards.
- **Model/client span** — An OTel span wrapping a single LLM API call. Carries `gen_ai.request.model`, token usage attributes, and optionally response metadata.
- **Agent span** — A span wrapping an orchestration loop (LangGraph, CrewAI, custom ReAct). Named `invoke_agent {gen_ai.agent.name}`. Parent of model and tool spans.
- **`OTEL_SEMCONV_STABILITY_OPT_IN`** — Environment variable that controls adoption of semantic convention versions. Set to include `gen_ai` to opt into the GenAI attribute schema and optional content capture.
- **Content capture** — The practice of logging raw prompt and completion text on GenAI spans. Defaults to off for privacy and storage reasons; opt-in via instrumentation configuration.
- **`pipeline.run_id`** — A custom (non-convention) correlation attribute you attach to every span in an enrichment run. Functions as a join key for cost attribution queries.

## Sources

- OpenTelemetry Semantic Conventions — Generative AI: `https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/README.md` (specification of `gen_ai.*` attributes, span categories, and content capture policy)
- OpenTelemetry Python SDK Documentation: `https://opentelemetry.io/docs/languages/python/` (TracerProvider, SpanProcessor, ConsoleSpanExporter API used in Build It and Use It)
- OpenTelemetry Semantic Convention Stability and Opt-In: `https://opentelemetry.io/docs/specs/semconv/` (documentation of `OTEL_SEMCONV_STABILITY_OPT_IN` environment variable and versioning policy)