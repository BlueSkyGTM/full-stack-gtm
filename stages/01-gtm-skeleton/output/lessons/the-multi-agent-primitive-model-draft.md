# The Multi-Agent Primitive Model

---

## Beat 1: Hook — When One Agent Isn't Enough

Single-agent architectures hit a capability ceiling: context window saturation, tool sprawl, and conflicting system prompts. The multi-agent primitive decomposes a task across specialized agents that hand off control and state. This is the same decomposition pattern behind GTM enrichment waterfalls — sequential providers, fallback logic, result aggregation — just at the agent level instead of the API level.

---

## Beat 2: Model — Agents, Edges, Topology

Three primitives define any multi-agent system: the **agent** (a unit of reasoning + tool access), the **edge** (a directed handoff with payload), and the **topology** (the graph shape — chain, fan-out, fan-in, or hierarchy). Every multi-agent framework implements these three. [CITATION NEEDED — concept: formal multi-agent topology taxonomy in production systems]

---

## Beat 3: Mechanism — Handoffs, Shared State, and Error Propagation

The hard problem isn't calling two agents — it's managing what passes between them. This beat covers: structured handoff payloads (not free-text hope), shared state stores vs. message-passing, and how failure in agent N propagates (or doesn't) to agent N+1. The mechanism that makes this work is a **supervisor** agent or a **state machine** that routes control based on the output of the previous step.

*Exercise hooks:*
- **Easy:** Build a two-agent chain where Agent A extracts a company name from raw text and passes it to Agent B, which looks up the domain. Print the full handoff payload.
- **Medium:** Implement a fan-out pattern — one supervisor sends the same input to three specialized agents (one per data provider), then aggregates results. Print latency per agent and final merged output.
- **Hard:** Build a supervisor that re-routes on failure: if Agent A returns an error, fall back to Agent B with a modified prompt. Log every routing decision.

---

## Beat 4: Use It — GTM Redirect: Enrichment Waterfalls as Multi-Agent Topology

**GTM Cluster: Zone 3 — Enrichment & Scoring.**

The Clay enrichment waterfall is a multi-agent chain: try provider A → if null, try provider B → if null, try provider C → merge results. Same topology, same fallback logic, same aggregation step. This beat maps the multi-agent primitive directly to the enrichment waterfall pattern, showing that once you can build a multi-agent chain, you can build (and debug) any waterfall in Clay or any enrichment orchestration layer.

*Exercise hooks:*
- **Easy:** Implement a three-step enrichment waterfall as a sequential agent chain. Each agent "calls" a different mock provider. Print the final enriched record and which provider filled each field.
- **Medium:** Add conditional logic — skip provider B if provider A returned a confidence score above 0.8. Print the routing decision and cost savings.
- **Hard:** Implement parallel fan-out enrichment (three providers simultaneously) with a merge step that resolves conflicts by confidence score. Output the final record with provenance metadata.

---

## Beat 5: Ship It — Observability, Cost, and the N² Problem

Every additional agent multiplies latency, token cost, and failure surface area. This beat covers: tracing handoffs with structured logs (who called whom, what payload, how long), cost accounting per agent, and the combinatorial explosion problem when agents can call each other cyclically. You'll instrument a multi-agent run to produce a trace log that shows exactly where time and tokens were spent.

*Exercise hooks:*
- **Easy:** Add timing and token-count logging to every handoff in a two-agent chain. Print a summary table.
- **Medium:** Instrument a three-agent system to detect and break cycles — if an agent hands off back to a previous agent, log a warning and terminate. Print the cycle detection event.
- **Hard:** Build a cost gate: before each handoff, check cumulative token spend against a budget. If over budget, return a partial result with a "budget_exhausted" flag instead of calling the next agent.

---

## Beat 6: Evaluate — Does Decomposition Actually Help?

Not every task benefits from multi-agent decomposition. This beat provides a decision framework: decompose when (1) sub-tasks require different tools, (2) context windows overflow, or (3) parallel execution yields real latency gains. Don't decompose when a single agent with good prompting solves it in one pass. The evaluation is empirical — measure latency, cost, and accuracy for both approaches on the same task.

*Exercise hooks:*
- **Easy:** Run the same enrichment task twice — once with a single agent, once with a three-agent chain. Print a comparison table: latency, tokens, accuracy.
- **Medium:** Build a benchmark harness that runs N tasks across both architectures and outputs aggregate stats (mean latency, p95 latency, mean cost, accuracy delta).
- **Hard:** Implement an adaptive router that chooses single-agent vs. multi-agent based on input complexity (e.g., number of fields to enrich). Log the routing decision and whether it was correct in hindsight.

---

## Learning Objectives

1. **Implement** a sequential multi-agent chain with structured handoff payloads between agents.
2. **Compare** three multi-agent topologies (chain, fan-out, hierarchy) and identify when each is appropriate.
3. **Build** an enrichment waterfall using the multi-agent chain pattern, with fallback logic and result aggregation.
4. **Instrument** a multi-agent system to log handoff payloads, per-agent latency, and cumulative token cost.
5. **Evaluate** whether multi-agent decomposition improves a given task along the dimensions of latency, cost, and accuracy.

---

## GTM Redirect Summary

| AI Concept | GTM Cluster | Redirect |
|---|---|---|
| Sequential agent chain | Zone 3 — Enrichment & Scoring | This is the enrichment waterfall pattern Clay implements |
| Fan-out / parallel agents | Zone 3 — Enrichment & Scoring | Parallel provider calls with merge — same as multi-provider enrichment |
| Supervisor / router agent | Zone 4 — Activation & Outreach | Orchestration of multi-step outreach sequences |
| Error propagation & fallback | Zone 3 — Enrichment & Scoring | Provider fallback logic in data enrichment pipelines |