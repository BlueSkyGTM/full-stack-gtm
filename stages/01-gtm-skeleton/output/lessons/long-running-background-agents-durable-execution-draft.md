# Long-Running Background Agents: Durable Execution

## 1. Hook

A web request dies after 30 seconds. Your agent needs 12 minutes to research a company, cross-reference three data sources, and draft a personalized outreach sequence. When the container restarts mid-run, everything vanishes. This lesson covers the execution model that prevents that.

## 2. Concept

Durable execution separates **progress** from **process**. The mechanism: every step in a workflow checkpoints its output to a durable store. On failure, the orchestrator replays from the last checkpoint—not from the start. Contrast this with naive retry (restart everything) and queue-based retry (restart the task, but lose internal progress). Key terms: workflow, activity, checkpoint, replay, saga pattern for compensation. Frameworks implementing this: Temporal (event-sourced replay), Inngest (step functions with automatic durability), LangGraph (checkpointed graph state). The critical trade-off: checkpointing adds latency per step, but converts catastrophic failure into minor delay.

## 3. Demo

Build a minimal durable workflow using Python that simulates a multi-step enrichment agent. Each step checkpoints its result. Kill the process mid-execution, restart it, and observe it resumes from the last completed step—not from scratch. Print checkpoint state at each stage so the observer can verify recovery. Use `sqlite3` as the durable store (no external dependencies). ~80 lines of working code.

## 4. Deep Dive

**Replay vs. Re-execute**: Temporal replays the event history to reconstruct state; a naive checkpoint system re-executes from the last checkpoint. The difference matters when steps have side effects (API calls). Solution: deterministic replay with recorded side-effect results.

**Idempotency requirement**: If a step completes but the checkpoint write fails, the step re-runs. Every activity must be safe to call twice with the same input. Pattern: use idempotency keys on external API calls.

**Compensation / saga**: When step 4 of 5 fails, steps 1–3 have already committed. The saga pattern defines a compensating action per step (undo). Without it, you have orphaned state.

**Cost of durability**: Each checkpoint is a write to disk/network. For a 50-step workflow, that's 50 writes. The overhead is ~5–20ms per step on local disk, ~50–200ms on network storage. This is acceptable for agent workflows (seconds per step) but prohibitive for hot-path request processing.

## 5. Use It — GTM Redirect

**Cluster: Zone 2 (Enrich) — Multi-Source Waterfall Workflows**

In Clay, a waterfall enrichment queries Provider A, then Provider B, then Provider C until a field is populated. Each provider call is a step. If the workflow crashes after Provider B returns data, durable execution ensures Provider B's result is retained and Provider C runs next—not re-querying A and B. The same pattern applies to Clay's enrichment waterfall: each cell's enrichment is a durable activity. Without checkpointing, a platform restart means re-fetching (and re-paying for) every data source. [CITATION NEEDED — concept: Clay's internal enrichment checkpointing behavior]

**Exercise (Easy)**: Write a 3-step enrichment simulation that checkpoints after each "API call." Kill and restart. Print which steps were skipped on recovery.

## 6. Ship It

**Exercise (Medium)**: Implement a durable 5-step agent workflow (research → enrich → score → draft → log) with SQLite checkpointing. Add idempotency keys to each step. Kill the process after step 3. Restart and confirm steps 1–3 are skipped, step 4 runs. Print the full execution log showing recovery.

**Exercise (Hard)**: Add saga-style compensation. If step 4 (draft) fails, automatically run compensating actions for steps 1–3 (e.g., mark the prospect record as "enrichment_failed" instead of leaving it in an inconsistent state). Log every compensation action. This is the pattern production enrichment systems use to avoid orphaned CRM records.

---

## Learning Objectives (draft)

1. **Implement** a checkpoint-based durable workflow that survives process restarts and resumes from the last completed step.
2. **Explain** the difference between replay-based durability (Temporal) and checkpoint-based durability, and the trade-offs of each.
3. **Diagnose** when a non-idempotent activity will cause incorrect behavior on replay, and fix it with idempotency keys.
4. **Implement** a saga-style compensation pattern that rolls back side effects when a multi-step workflow fails mid-execution.
5. **Evaluate** whether a given agent workflow needs durable execution or whether simpler retry semantics suffice, based on step count, side-effect cost, and failure probability.