# Failure Modes: Why Agents Break

## Hook
Agents don't fail randomly — they fail in predictable, repeating patterns. This beat maps the six failure modes that account for most production agent breakage: loop exhaustion, tool parameter drift, context window overflow, hallucination cascades, state loss, and silent degradation.

## Concept
Define each failure mode by its mechanism: what triggers it, what the agent does wrong, and what the observable symptom looks like in logs or output. Introduce the "failure taxonomy" — a classification scheme that lets you diagnose root cause from symptom in under 60 seconds.

## Mechanism
Walk through three failure modes in depth using execution traces. Show the exact point where the agent's reasoning diverges from intended behavior. For each: the prompt state, the tool call, the model's completion, and where the failure cascades. Emphasize that most agent failures are compounding — one small drift creates the next.

## Code
Build a minimal agent loop with instrumentation that detects each failure mode in real-time. The code runs a simulated multi-step task, injects a controlled failure (bad tool response, context overflow), and prints the diagnostic output showing which failure mode tripped and at which step.

## Use It
GTM redirect: enrichment waterfalls and multi-step sequence agents in Clay and similar platforms exhibit these exact failure modes. When a Clay waterfall stalls on a row, or a sequence agent sends a malformed email, the root cause maps to one of the six failure modes. Diagnosing by taxonomy replaces "it just stopped working" with actionable root cause. [CITATION NEEDED — concept: Clay enrichment waterfall failure diagnostics]

## Ship It
Exercise hooks:
- **Easy**: Run the instrumented agent code on three provided scenarios. Classify each failure by taxonomy category. Print the diagnosis.
- **Medium**: Add a fifth failure mode — rate limit handling — to the instrumented agent. Write a detection function that identifies it from the execution trace.
- **Hard**: Build a recovery mechanism for one failure mode. When the detector trips, the agent retries with a corrected prompt or escalates. Demonstrate the agent recovering from an injected failure and completing the task.

---

## Learning Objectives (Draft)

1. Classify an agent failure into one of six taxonomy categories given an execution trace.
2. Diagnose the root cause of an agent loop stall from log output.
3. Implement instrumentation that detects tool parameter drift in a running agent.
4. Compare hallucination cascade vs. state loss using observable symptoms.
5. Build a detection function that identifies context window overflow before it crashes the agent.