# ICM Hierarchy — Getting Started

Entry point for any agent or operator new to the BlueSkyGTM curriculum pipeline.

**As of 2026-06-14 this pipeline is GLM-native.** The Taskmaster tier moved out of Claude sub-agents and into GLM-5.2. The whole Taskmaster→Handler chain now runs inside one GLM ecosystem — **The Loom**: GLM-5.2 sets the pattern, GLM-5.1/5.1v are the threads, and every lesson is woven from the AI strand and the GTM strand at once. The Conductor (Claude) launches The Loom and watches the manifest — it no longer holds a sub-agent per stage.

> **The Loom** = the GLM collective that builds the curriculum (GLM-5.2 + GLM-5.1 + GLM-5.1v). ICM/ICL is *how it's organized*; The Loom is *the network itself*. The Conductor conducts; The Loom weaves.

---

## The ICM Pattern

ICM = Interpreted Context Manifest. Every agent orients itself by reading its environment, not by receiving a verbal briefing:
- **CONTEXT.md** — what this agent owns, reads, writes, and how it escalates
- **manifest.json / status.json** — the map: every work item with id, status, input, output
- The upstream trigger (a path handed to the agent) **is** the instruction

This is what makes the system re-entrant: kill any process, restart it, point it at the same folder — it reads the manifest and continues exactly where the last one stopped.

---

## The Stack (GLM-native, 4 tiers)

```
Director (human)
  — intent, quality gates, stage advancement. Never a terminal.

  └─ Conductor (Claude Code)                         ← you are here
       — architecture, gates, commits, stage chaining, Director comms
       — launches the GLM orchestrator, monitors the manifest
       — ~15% of token spend. Holds NO sub-agent per stage anymore.

       └─ GLM-5.2 Taskmaster  (single in-loop overseer, 1M context)
            — reads the whole manifest + batch plan + handler outputs in one window
            — dispatches Handler calls, judges each output against the lean spec,
              decides done / retry / failed, updates the manifest, writes status.json
            — IS the quality gate (reasoning-judged, not just regex)

            └─ GLM-5.1 / GLM-5.1v Handlers   (content writers)
                 — 5.1: lessons, exercises, quizzes (text)
                 — 5.1v: illustrations, diagrams (vision)
                 — write to spec; escalate a rejected output back to the Taskmaster
```

**Peons (GLM-5-Turbo)** are now optional. GLM-5.2's reasoning + context absorbs most interrupt-resolution (citation judgment, format checks) inline. Keep `dispatch-peon.py` only for the cheapest high-volume synchronous lookups.

---

## Why GLM-5.2 Changed the Architecture

| GLM-5.2 capability | What it unlocked |
|--------------------|------------------|
| **1,000,000-token context** (`glm-5.2[1m]`) | A GLM model can finally hold a whole batch + the manifest + outputs at once — so the overseer role no longer needs Claude's context. The Taskmaster moved into GLM. |
| **131,072 max output tokens** | The truncation bug ceases to exist. The 8K ceiling that guillotined 270 lessons is now 131K — no lesson reaches it. |
| **Enhanced reasoning (max effort)** | GLM-5.2 makes the dispatch/retry/escalation/quality-gate calls that previously required Claude judgment. |
| **$0 / Coding-Plan-covered** | Pushing 85% of token spend onto GLM has no per-token cost. Free to delegate aggressively. |

**Endpoint (all GLM tiers):** `https://api.z.ai/api/coding/paas/v4`
**Model strings:** `glm-5.2` (taskmaster) · `glm-5.2[1m]` (taskmaster, max-context oversight) · `glm-5.1` (text handler) · `glm-5.1v` (vision handler) · `glm-5-turbo` (optional peon). The endpoint is case-tolerant; prefer lowercase going forward.

---

## What This Frees (the point of the restructure)

- **Conductor context window** — no sub-agent transcripts accumulating per stage; the manifest is the only state I read.
- **Claude sub-agent slots** — freed for actual conductor work (architecture, review, the Loop-1 build), not babysitting dispatchers.
- **Claude token budget** — ~85% of generation spend is now GLM (plan-covered $0); Claude spends only on orchestration and gates.

---

## Quick Start for Conductor

### Launch the GLM orchestrator for a stage (Conductor does this directly via Bash):
```powershell
.\run.ps1 stage02 --sample 5      # human gate: review 5 before full run
.\run.ps1 stage02                  # full run — GLM-5.2 oversees, GLM-5.1 writes
.\run.ps1 status                   # manifest state for all stages
```
No Agent-tool sub-agent is spawned. The orchestrator is a GLM-native process; the Conductor monitors `manifest.json` / `status.json`.

### Check pipeline state:
```powershell
python -c "import json; rows=json.load(open('stages/02-lesson-injection/output/manifest.json',encoding='utf-8')); s={};[s.__setitem__(r['status'],s.get(r['status'],0)+1) for r in rows]; print(s)"
```

---

## Worker Budget (global)

**Max 5 concurrent GLM Handler calls.** (Z.ai degrades hard past this — 63% failure observed at 8.) The GLM-5.2 Taskmaster oversight call does not count against the Handler budget; it is one large-context call per batch cycle, not per item.

---

## Escalation in One Line Per Tier

- **Handler output rejected by the gate** → Taskmaster retries (≤3) → marks row `failed`
- **GLM-5.2 Taskmaster can't resolve a batch** → writes BLOCKED to status.json → Conductor reads it
- **Conductor blocked** → one message to Director: what's blocked, what was tried
- **Watchdog** → stall (no manifest progress in 5 min) → graceful stop → restart `--retry-failed`

---

## Key Files

| File | Purpose |
|------|---------|
| `HANDLERS.md` | Full registry: every tier, model, config, failure mode |
| `TASKMASTER-PROTOCOL.md` | The GLM-5.2 orchestrator pattern (replaces the Claude-sub-agent launcher) |
| `interpreted-context-manifest/SKILL.md` | The ICM pattern |
| `interpreted-context-loop/SKILL.md` | The ICL loop pattern |
| `watchdog.py` | Stall monitor — Conductor runs this |
| `taskmasters/<role>/CONTEXT.md` | Per-stage orchestrator brief |

---

## → Next: run it

The spine ends here, on the file that does the work:

```bash
export $(grep -v '^#' .env | xargs)                          # load ZHIPUAI_API_KEY
python skills/operator-kit/orchestrator.py --dry-run --sample 5   # confirm work selection
python skills/operator-kit/orchestrator.py --workers 5            # fire The Loom
```

`orchestrator.py` reads the manifest, lets GLM-5.2 oversee GLM-5.1, and weaves until `pending = 0`. Monitor the manifest; don't weave yourself.
