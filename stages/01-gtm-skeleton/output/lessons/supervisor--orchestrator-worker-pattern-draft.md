# Lesson: Supervisor / Orchestrator-Worker Pattern

## Beat 1: Hook

A single agent trying to do everything produces incoherent output. The supervisor pattern splits "decide what to do" from "do the work"—one agent routes, specialized agents execute.

## Beat 2: Core Concept

Define the architecture: one orchestrator LLM call decomposes a task, assigns subtasks to worker agents (which may themselves be LLM calls, API functions, or deterministic code), collects results, and synthesizes output. Contrast with swarm/mesh patterns where peers negotiate directly.

## Beat 3: Mechanism

Explain the control flow: task decomposition → worker selection → parallel or sequential execution → result aggregation → final synthesis. Cover failure modes (worker timeout, orchestrator routing errors, context window overflow from collecting too many worker outputs).

## Beat 4: Code Foundation

Build a minimal supervisor-worker system from scratch using the Anthropic API. Orchestrator receives a task, returns structured JSON naming which workers to invoke and with what inputs. Workers execute. Orchestrator synthesizes. Observable output: print the decomposition plan, each worker's result, and the final synthesis.

**Exercise hooks:**
- *Easy:* Add a new worker type to an existing supervisor loop.
- *Medium:* Implement retry logic when a worker returns an error.
- *Hard:* Add parallel worker execution with async and measure speedup vs sequential.

## Beat 5: Use It

Map to GTM enrichment workflows: an orchestrator agent receives a company name, decomposes into subtasks (domain lookup, employee count, tech stack detection, funding check), routes each to a specialized worker, and assembles a consolidated account record. This is the same decomposition pattern used in multi-source enrichment waterfalls [CITATION NEEDED — concept: enrichment waterfall orchestration in GTM tooling].

**Exercise hooks:**
- *Easy:* Define worker specializations for a lead-scoring decomposition.
- *Medium:* Build an orchestrator that takes a raw company name and outputs a structured enrichment record using three workers.
- *Hard:* Add confidence scoring—orchestrator evaluates worker outputs and re-routes low-confidence results to an alternate worker.

## Beat 6: Ship It

Production concerns: orchestrator prompt drift causing bad routing, worker latency budgets, token cost of passing full worker context back to the orchestrator, and observability (you need logging at the orchestration layer, not just the worker layer). When to abandon this pattern in favor of hardcoded pipelines (answer: when task decomposition is deterministic, skip the LLM orchestrator and use code).

**Exercise hooks:**
- *Easy:* Add timing instrumentation to each worker call and print a summary.
- *Medium:* Implement a token budget guard that aborts if accumulated worker outputs exceed a threshold.
- *Hard:* Replace the LLM orchestrator with a deterministic router for a fixed set of known task types; compare cost and accuracy.

---

## Learning Objectives (for `docs/en.md`)

1. **Implement** a supervisor-worker system that decomposes a task, dispatches to specialized workers, and synthesizes results.
2. **Diagnose** orchestrator routing failures by inspecting decomposition output.
3. **Compare** supervisor-worker architecture to hardcoded pipelines on cost, latency, and flexibility axes.
4. **Instrument** a multi-agent loop with timing and token-usage observability.
5. **Evaluate** when task decomposition is deterministic enough to replace an LLM orchestrator with code.