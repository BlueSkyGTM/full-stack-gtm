# Taskmaster Protocol — GLM-5.2 Orchestrator Pattern

**Changed 2026-06-14.** Taskmasters are no longer Claude sub-agents spawned via the Agent tool. They are **GLM-5.2 orchestration processes** the Conductor launches via Bash. The oversight that used to cost Claude context now runs inside the GLM ecosystem.

Old pattern (deprecated): Conductor → Agent tool → Claude sub-agent (thin launcher) → polls status.json → reports back.
New pattern: Conductor → `run.ps1 <stage>` → GLM-5.2 orchestrator → oversees GLM-5.1 Handlers → writes manifest → exits.

---

## Why It Changed

A Claude sub-agent was the Taskmaster because GLM-5.1 couldn't hold a batch plan + the manifest + handler outputs + the spec to judge against — its context was too small to *oversee*. GLM-5.2's 1M-token window removes that limit. The overseer can now be GLM. Result: one GLM ecosystem (no hybrid handoff), Claude's context freed, 85% of spend on plan-covered GLM.

---

## The Pattern

```
Conductor (Bash):  .\run.ps1 stage02      # or: python skills/operator-kit/dispatch-stage02.py
                   └─ launches the GLM-5.2 orchestrator for that stage's CONTEXT.md

GLM-5.2 Taskmaster (one process, large-context):
  1. Reads taskmasters/<role>/CONTEXT.md  → what to run, spec to judge against, output paths
  2. Reads manifest.json                  → all pending/failed rows (holds them in 1M context)
  3. Plans the batch                      → order, retry budget, concurrency (≤5 handlers)
  4. Loop until pending = 0:
       a. Dispatch ≤5 GLM-5.1 Handler calls (text) / GLM-5.1v (vision)
       b. Judge each returned output against the lean spec — the TWO-TIER GATE:
            structure-complete (fences balanced, required headers, per-section min-chars)
            ship-ready        (zero [CITATION NEEDED], weave names the AI mechanism)
       c. done → write output + mark done | reject → re-prompt (≤3) | exhausted → mark failed
       d. Update manifest.json + status.json
  5. pending = 0 → write finished:true → exit
```

The Conductor does **not** sit in this loop. It launches and monitors `manifest.json` at conversation breaks (per ICM active-monitoring cadence).

---

## Conductor Launch (replaces the Agent-tool prompt)

```powershell
# Foreground gate sample:
.\run.ps1 stage02 --sample 5
# Full run, background, Conductor monitors the manifest:
python skills/operator-kit/dispatch-stage02.py --workers 5    # run_in_background
```

There is no Agent-tool sub-agent prompt anymore. The CONTEXT.md + the manifest are the full instruction set for the GLM-5.2 orchestrator.

---

## GLM-5.2 Orchestrator — Reference Implementation (build target)

The orchestrator wraps the existing handler-dispatch loop with a GLM-5.2 oversight layer:

```python
# skills/operator-kit/orchestrator.py  (Phase-B build target)
#
# config: { stage, manifest_path, spec_path, output_root, workers<=5 }
#
# TASKMASTER_MODEL = "glm-5.2"        # 1M context, 131K output, max-effort reasoning
# HANDLER_MODEL    = "glm-5.1"        # text;  "glm-5.1v" for vision
# ENDPOINT         = "https://api.z.ai/api/coding/paas/v4"
#
# 1. load manifest + lean spec (spec held in the 5.2 system context for every judge call)
# 2. while pending:
#      batch = next <=5 pending rows
#      outputs = parallel_handler_calls(batch, HANDLER_MODEL, max_tokens=32000)  # log finish_reason
#      verdicts = glm52_judge(outputs, spec)        # one large-context 5.2 call judges the batch
#      for row, verdict in verdicts:
#          if verdict.ship_ready:      write(row); mark(row,"done")
#          elif verdict.retryable and row.attempts<3:  mark(row,"pending")   # re-prompt with verdict notes
#          else:                       mark(row,"failed", reason=verdict.gap)
#      save_manifest(); save_status()
# 3. write finished:true; exit
```

Two design rules:
- **The 5.2 judge call is batched, not per-item** — judge ≤5 outputs in one large-context call. That keeps oversight to one call per cycle and uses the 1M window for what it's for.
- **The spec lives in the 5.2 system context, not in each handler prompt.** Handlers stay on the governed-maze ≤500-token extract; the *judge* holds the full spec. Oversight knowledge concentrates in the overseer.

---

## Worker Budget

**≤5 concurrent GLM-5.1 Handler calls.** The GLM-5.2 Taskmaster oversight call is separate (one per batch cycle) and does not count against the 5. Running two stages' orchestrators in parallel still shares the 5-handler ceiling — split workers (e.g. 2+3) or serialize.

---

## Re-entrancy (ICL guarantee)

Kill the orchestrator mid-run. Relaunch it. It reads the manifest, finds the surviving `pending`/`failed` rows, and continues — no session memory needed. The manifest is the loop counter; the 5.2 orchestrator is stateless across restarts. This is ICL working inside a single GLM ecosystem.

---

## Escalation

| Event | Orchestrator does |
|-------|-------------------|
| Handler output rejected 3x | Mark row `failed` with the gap reason from the 5.2 judge |
| 429 / rate limit | Global backoff 30s, all handlers pause |
| >30% failure in 10 jobs | Circuit breaker, pause 60s |
| Batch unresolvable | Write `BLOCKED` + reason to status.json; Conductor reads it |
| Stall (no manifest progress 5 min) | Watchdog graceful-stops and relaunches `--retry-failed` |
