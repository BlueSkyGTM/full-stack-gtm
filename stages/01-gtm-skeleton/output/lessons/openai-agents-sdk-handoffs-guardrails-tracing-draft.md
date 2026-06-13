# OpenAI Agents SDK: Handoffs, Guardrails, Tracing

## Hook
You're building a multi-step research agent that qualifies leads. Halfway through, it needs to switch from "research mode" to "evaluate mode." You also need to prevent it from hallucinating company data, and you need to see *why* it flagged a lead as high-fit. Handoffs, guardrails, and tracing are the three primitives that solve these problems in the OpenAI Agents SDK.

## Concept
Three mechanisms, each solving a distinct control-flow problem in multi-agent systems:

- **Handoffs**: A control-flow primitive where Agent A transfers execution to Agent B along with conversation state. Not a function call—the calling agent terminates and Agent B becomes the active agent. The SDK implements this as a special tool the first agent can invoke, which serializes context and re-instantiates a new agent runner.

- **Guardrails**: Synchronous or async validation functions that intercept agent input or output. Input guardrails run before the LLM call; output guardrails run after. A guardrail returns a `GuardrailResult` with `tripwire_triggered`—if true, the agent execution halts or the output is rejected. This is the SDK's mechanism for enforcing boundaries without baking them into the prompt.

- **Tracing**: Structured event emission during agent execution. The SDK auto-instruments spans for each agent invocation, tool call, and handoff. Traces are hierarchical—each span has a parent. You can export these to OpenAI's trace dashboard or consume them programmatically via `trace.process_span`. This is observability, not logging: you get causal chains, not timestamped text.

## Demonstration
Working code showing all three primitives in a single runnable script:
- Define two agents (research agent, evaluation agent) with a handoff from research → evaluation
- Add an output guardrail on the research agent that rejects responses containing fabricated revenue numbers
- Enable tracing and print the trace spans to stdout to show the full execution path
- Output: observable trace showing handoff event, guardrail check, and final result

Exercise hooks:
- **Easy**: Modify the guardrail to reject a different pattern (e.g., any output containing "estimated")
- **Medium**: Add a third agent and a conditional handoff (research → evaluation OR research → enrichment based on output content)
- **Hard**: Write a custom trace processor that counts handoffs and guardrail triggers, then asserts the counts match expected values

## Use It
**GTM Redirect**: This maps to the multi-agent enrichment pipeline in Zone 2. A real GTM workflow: research agent enriches a domain, guardrails reject outputs with placeholder data, handoff moves to a scoring agent, and tracing lets you debug *why* certain leads scored high.

Specific application: Replace a Clay waterfall with an agent pipeline where each step is a specialist agent. Guardrails enforce data quality before writing to your CRM. Tracing gives you audit trails for compliance.

Exercise hooks:
- **Easy**: Build a two-agent pipeline that researches a company domain then scores ICP fit, with a handoff between them
- **Medium**: Add an output guardrail that rejects enrichment data missing employee count or industry
- **Hard**: Instrument the pipeline with tracing, export spans, and calculate what percentage of runs complete without tripping a guardrail

## Ship It
Production concerns for each primitive:

- **Handoffs**: Serialize only what the next agent needs. Full conversation history across 5 handoffs = token bloat. Implement `context_filter` functions that strip state during transfer.
- **Guardrails**: Run input guardrails in parallel (the SDK supports this). Set hard timeouts—a guardrail that calls an external API can block the entire pipeline.
- **Tracing**: Sample traces in production (not every run needs full instrumentation). Export to your existing observability stack via custom processors, not just the OpenAI dashboard.

Exercise hooks:
- **Easy**: Add a timeout to a guardrail function and verify it fails gracefully
- **Medium**: Write a context filter that strips all but the last 3 messages before a handoff, measure token reduction
- **Hard**: Build a custom trace exporter that writes spans to a local JSONL file with a configurable sample rate

## Own It
The SDK implements handoffs as tool calls under the hood—inspect `agent.tools` after registering a handoff and you'll see the synthetic tool. Guardrails are advisory, not architectural: a determined prompt can work around them, so treat them as defense-in-depth, not security boundaries. Tracing adds latency; benchmark with it on and off before shipping.

Deeper directions: implement dynamic agent selection at handoff time (not hardcoded), chain guardrails into a validation pipeline with short-circuit evaluation, build a trace replay system that lets you re-execute a failed run from any span.

Exercise hooks:
- **Easy**: Print the synthetic handoff tool schema the SDK generates and identify its fields
- **Medium**: Build a guardrail that calls a local validation function and compare its latency to a pure-regex guardrail
- **Hard**: Implement a minimal trace replay: given a saved trace, re-invoke each agent with the same inputs and verify outputs match within a threshold