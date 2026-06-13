# Plan: Agent Hierarchy Redesign

## Problem

The Director (user) is typing terminal commands. That is a delegation failure.
The curriculum pipeline stalls and the Director notices it. That is a monitoring failure.
There is no documented chain of command. That is a design failure.

The curriculum is the product. Infrastructure serves it. Everything else is scaffolding.

## Goal

Build a five-tier agent hierarchy where the Director gives intent and receives results.
Nothing in between requires Director input unless it is a quality gate or stage advancement decision.

## The Hierarchy

```
Director (User)
  — intent, quality gates, stage advancement only

  └─ Conductor (Claude Code)
       — translates intent to plans, keeps big picture, delegates downward
       — escalates to Director: quality gates, stage advancement, billing decisions

       └─ Taskmasters (Claude sub-agents, Anthropic models)
            — spawned by Conductor for a scoped job (one stage, one phase)
            — reads CONTEXT.md from its folder, launches Python dispatch subprocess
            — when subprocess completes: reports to Conductor and exits (not persistent)

            └─ Handlers (GLM-5.1 — primary content writers)
                 — write lessons, exercises, quizzes — this is their main job
                 — when a menial interrupt would break flow: delegate to Peon
                 — after Peon resolves interrupt: continue writing
                 — escalate to Taskmaster only on persistent failure

                 └─ Peons (GLM-5-Turbo / GLM-Flash — interrupt resolvers)
                      — invoked synchronously by Handler mid-work
                      — resolve a specific subtask (citation, validation, metadata)
                      — return result to Handler and exit
                      — never run in background; never accumulate state
```

## What Gets Built

### 1. HANDLERS.md — Chain of Command Registry
Centralized documentation for every handler in the system.
Each entry: what it does, who invokes it, what it reads, what it writes, failure modes, escalation path.
This is the single source of truth for the hierarchy.

Location: `skills/operator-kit/HANDLERS.md`

### 2. Watchdog Handler
A script the Conductor runs (not the Director) that monitors dispatcher health.
- Polls `status.json` every 60 seconds
- Detects stall: no progress in 5 minutes
- Auto-recovery: kills hung process, restarts with `--retry-failed`
- Writes recovery log that Conductor reads
- Only surfaces to Director when auto-recovery fails 3x

Location: `skills/operator-kit/watchdog.py`

### 3. Taskmaster Sub-Agent Protocol
A documented pattern for how the Conductor spawns Claude sub-agents as Taskmasters.
- Each Taskmaster gets a scoped brief: one stage, one phase, one responsibility
- Taskmaster monitors Handler output, validates quality, reports back to Conductor
- Conductor synthesizes Taskmaster reports, presents summary to Director

Location: `skills/operator-kit/TASKMASTER-PROTOCOL.md`

### 4. Peon Dispatcher Architecture
Lower GLM models handle high-volume simple tasks, managed by GLM-5.1 Handlers.
Peon use cases:
- Citation lookup (GLM-Flash scanning sources for a lesson)
- Format validation (checking heading structure, JSON schema)
- Simple text extraction (pulling learning objectives from a full lesson)
- Metadata generation (tags, difficulty ratings, strand labels)

GLM-5.1 Handler constructs the peon prompt, calls GLM-Flash, validates output.
If peon output fails validation twice, Handler escalates to Taskmaster.

Location: `skills/operator-kit/dispatch-peon.py` (shared peon caller)

### 5. Delegation Protocol
Explicit rules for what each tier decides autonomously vs escalates up.

| Decision | Tier that decides |
|----------|------------------|
| Worker count adjustment | Conductor |
| Retry failed rows | Conductor |
| Stall recovery | Watchdog (auto) |
| Stage advancement | Director |
| Quality gate pass/fail | Director (with Conductor summary) |
| Model selection within tier | Conductor |
| New stage architecture | Director |
| API key rotation | Director |

## Design Principles

1. **Director gives intent, receives results.** No terminal commands, no stall notifications.
2. **Same-family model management.** GLM models managed by GLM-5.1, not Anthropic.
3. **Curriculum first.** Every infrastructure decision is justified by curriculum output quality.
4. **Explicit escalation paths.** Every handler knows its three failure modes and who to tell.
5. **Conductors keep the score.** I hold the big picture so Taskmasters can focus on execution.

## Corrected Mental Model (critical)

**Handlers are writers, not supervisors.**

A Handler (GLM-5.1) is generating content — a lesson, exercise, or quiz. Mid-generation it hits a blocking subtask that would interrupt its flow (citation lookup, schema validation, metadata tag generation). It delegates that interrupt to a Peon via dispatch-peon.py, gets the result back, and continues writing. The Peon is a synchronous interrupt resolver, not a subordinate running in parallel.

**Agents are invoked in stages, not running simultaneously.**

```
Conductor activates one Taskmaster at a time (or N for parallel phases)
  Taskmaster launches a Python dispatch subprocess
  That subprocess calls GLM-5.1 Handlers (3-5 concurrent, within budget)
    Each Handler writes content
    When Handler needs a menial interrupt resolved → calls dispatch-peon.py
    dispatch-peon.py calls GLM-Flash → returns result → Handler continues
  Taskmaster subprocess completes → Taskmaster reports to Conductor → Taskmaster dies
Conductor activates next Taskmaster (or presents summary to Director)
```

The system is never "fully active" — it's a pipeline of staged activations. Worker budget is respected because only one tier is actively calling the API at a time. Peon calls happen inside Handler calls (nested, synchronous), not in parallel.

**Each folder is a tool + knowledge package.** The CONTEXT.md in a folder IS the agent's prompt — it tells the agent what it owns, what tools are available to it, where inputs come from, where outputs go. The trigger from an upstream agent IS the instruction. The folder is everything the agent needs to execute without further input from the Director.

## ICM Architecture Clarification (post-review)

Taskmasters are thin subprocess launchers, not heavy LLM orchestrators. Pattern:

```
Conductor → spawns N Claude sub-agents (Taskmasters) via Agent tool
  Each Taskmaster reads: skills/operator-kit/taskmasters/<task>/CONTEXT.md
  Each launches: Python dispatch subprocess (detached, survives session end)
  Each monitors: status.json polling (NOT accumulating lesson content in context)

  Python dispatcher → GLM-5.1 API calls (governed maze, ≤500-token brief)
    GLM-5.1 Handler → calls dispatch-peon.py for simple subtasks
      GLM-Flash Peon → writes to subfolder, Handler integrates
```

**Critical constraints from engineering review:**
- Global worker budget: max 5 concurrent GLM calls TOTAL across all Taskmasters
- Per-Taskmaster manifests (avoid cross-process write contention); merge at end
- All output writes: `.tmp` → rename (atomic; survives watchdog kill)
- Watchdog: write initial status.json on launch; PAUSE_SENTINEL before kill
- MODEL_REGISTRY dict: model names in one place, not hardcoded everywhere
- dispatch-peon.py: validate ALL task dict fields against allowlist (injection prevention)

## Implementation Order (revised)

1. Write HANDLERS.md (chain of command + ICM folder structure documented)
2. Fix circuit breaker in existing dispatchers (use `_global_pause` event, not main-thread sleep)
3. Fix output file writes to use tmp→rename in all dispatchers
4. Write watchdog.py (PAUSE_SENTINEL-first, initial status.json, absent-file detection)
5. Write dispatch-peon.py (MODEL_REGISTRY, task dict validation, --smoke-test flag)
6. Write TASKMASTER-PROTOCOL.md (thin launcher pattern, per-Taskmaster manifest, detached process)
7. Create taskmasters/<task>/CONTEXT.md folder structure
8. Write HIERARCHY.md (1-page ICM getting-started)
9. Run Stage 02 full pipeline under new hierarchy (Conductor runs it, not Director)

## Success Criteria

- Director has not typed a terminal command since hierarchy went live
- Stage 02 full run completes without Director intervention
- Stall recovery is automatic and logged
- HANDLERS.md is the single doc any new agent needs to understand the system
- `dispatch-peon.py --smoke-test` passes before any peon-tier deployment
- No cross-process manifest write contention (per-Taskmaster manifests verified)

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|---------|
| 1 | CEO | Full hierarchy (five tiers) over simplified three-tier | Taste → confirmed by user | P1 completeness | ICM pattern makes five tiers architecturally meaningful, not ceremonial | Three-tier flatten |
| 2 | CEO | Defer "update dispatchers to use Peons" to after dispatch-peon.py validated | Mechanical | P3 pragmatic | Validate standalone first; don't touch production dispatchers until pattern proven | Ship together |
| 3 | CEO | HANDLERS.md + CONTEXT.md folders = machine-readable truth (not just static doc) | Mechanical | P5 explicit | Agents consume CONTEXT.md folders; HANDLERS.md is the index | Pure markdown registry |
| 4 | Eng | Global worker semaphore across Taskmasters | Mechanical | P1 completeness | 4x4=16 concurrent GLM calls exceeds safe ceiling; shared budget is the only fix | Per-Taskmaster ThreadPoolExecutor |
| 5 | Eng | Per-Taskmaster manifests, merge at end | Mechanical | P5 explicit | Cross-process tmp→rename is atomic per OS but last-writer-wins on concurrent writes to same file | Single shared manifest |
| 6 | Eng | Taskmasters are thin subprocess launchers, not LLM orchestrators | Mechanical | P5 explicit | LLM context overflow if Taskmasters accumulate lesson content; Python process is the right unit | Heavy LLM Taskmaster |
| 7 | Eng | Fix circuit breaker to use `_global_pause` event | Mechanical | P1 completeness | Current impl pauses main thread but workers keep calling GLM; event pauses all workers | Leave as-is |
| 8 | Eng | Watchdog uses PAUSE_SENTINEL before kill | Mechanical | P1 completeness | Avoids truncated output files; gives in-flight API call 30s to drain | Hard kill immediately |
| 9 | DX | MODEL_REGISTRY dict for model names | Mechanical | P4 DRY | "GLM-5.1" hardcoded in 15+ places; one version change requires grep-and-replace | Hardcoded strings |
