# Agent Observability: Langfuse, Phoenix, Opik

## Learning Objectives

- Instrument a multi-step agent with decorator-based tracing that captures parent-child span hierarchies, token counts, latency, and cost.
- Compare the trace-capture mechanisms of Langfuse (decorator-first), Phoenix (session-based with OpenInference auto-instrumentation), and Opik (eval-attached tracing).
- Compute cost-per-action and p95 latency from trace data and correlate them to GTM campaign metrics like cost-per-email-sent.
- Configure a local-to-production observability pipeline spanning Phoenix embedded, Langfuse self-hosted, and production backend routing.
- Diagnose a failing agent step from span-level token, latency, and error data within a trace tree.

## The Problem

You shipped an agent. It worked in testing. Now a user reports it generated a nonsensical email to a Fortune 500 CTO, and you have no idea what happened. Your logs show a 200 response from the LLM API. The agent returned a string. Everything looks green. The problem is somewhere between the research step, the retrieval step, and the draft step — and you cannot see inside.

That gap between "it returned 200" and "it produced the right output" is where observability lives. Traditional APM tools (Datadog, New Relic) capture HTTP request latency and error rates. They were built for deterministic services: a request comes in, a database query runs, a response goes out. Agents break this model. An agent makes multiple non-deterministic LLM calls, branches based on model output, invokes tools, and consumes tokens at variable rates. A single user request might produce five LLM calls costing $0.03 or fifteen calls costing $0.30, and the difference is invisible without instrumentation designed for the agent execution model.

The specific failure modes you need to catch: a retrieval step returning empty context (silent failure, agent hallucinates), a tool call timing out and the agent retrying three times (cost spike), a prompt template getting swapped without version tracking (quality regression), or a model upgrade changing output format (downstream parsing breaks). None of these show up in standard error logs because none of them produce errors — they produce wrong answers at normal HTTP status codes.

## The Concept

Every agent run produces a **trace**: the complete execution record from input to final output. Inside that trace, each discrete operation — an LLM call, a tool invocation, a retrieval query, a sub-agent delegation — becomes a **span**. Spans have parent-child relationships: the top-level agent span is the parent, and each step it calls is a child. A child span can itself be a parent if it triggers sub-operations (an LLM call that triggers a function call that triggers another LLM call). Each span carries **metadata**: token counts (input and output), latency, cost, model version, prompt template version, and arbitrary key-value tags you attach for filtering.

The mechanism for capturing this data follows one of three patterns. **Decorator-based wrapping** puts a `@trace` or `@observe` decorator on each function, and the tracing library uses Python's `ContextVar` to maintain the parent-child relationship automatically — when a decorated function calls another decorated function, the inner span knows its parent because the outer span's ID is in the context. **Callback-based integration** hooks into the LLM client's callback system (LangChain callbacks, OpenAI client wrappers) and fires on every API call, creating spans without modifying agent code. **Proxy interception** routes all LLM API traffic through a sidecar that captures requests and responses transparently — useful when you cannot modify the agent code but can control its network path.

```mermaid
flowchart TD
    TRACE["Trace: outbound_agent(company='Acme')"] --> SPAN1["Span: research_company\nkind=llm\nmodel=gpt-4o\ntokens_in=80\ntokens_out=200\ncost=$0.0022\nlatency=340ms"]
    TRACE --> SPAN2["Span: search_contacts\nkind=tool\ntool=web_search\nquery='Acme CTO'\nlatency=120ms"]
    TRACE --> SPAN3["Span: draft_email\nkind=llm\nmodel=gpt-4o\ntokens_in=280\ntokens_out=300\ncost=$0.0037\nlatency=510ms"]
    SPAN3 --> SPAN3a["Span: refine_email\nkind=llm\nmodel=gpt-4o\ntokens_in=350\ntokens_out=280\ncost=$0.0039\nlatency=480ms"]
    
    STYLE_TRACE fill:#1a1a2e,stroke:#e94560,color:#fff
    STYLE_SPAN1 fill:#16213e,stroke:#0f3460,color:#fff
    STYLE_SPAN2 fill:#16213e,stroke:#0f3460,color:#fff
    STYLE_SPAN3 fill:#16213e,stroke:#