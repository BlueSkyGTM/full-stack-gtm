# Handler Registry — Chain of Command (GLM-native, 2026-06-14)

Every agent in this system is documented here. Find your tier, understand your scope, follow your escalation path.

**Architecture change (2026-06-14):** the Taskmaster tier is now **GLM-5.2**, not Claude sub-agents. The Conductor launches a GLM-native orchestrator and monitors the manifest. See `TASKMASTER-PROTOCOL.md` for the new pattern.

---

## The Hierarchy

```
Director (User)
  — intent, quality gates, stage advancement only. Never types terminal commands.

  └─ Conductor (Claude Code)
       — reads Director intent, plans execution, launches the GLM orchestrator
       — owns: vault/plan-*.md, run.ps1, this file, the ICM artifacts
       — monitors manifest.json / status.json (no per-stage sub-agent)
       — escalates to Director: stage advancement, quality failures
       — ~15% of token spend

       └─ GLM-5.2 Taskmaster (in-loop overseer — single GLM ecosystem)
            — reads the full manifest + batch plan + handler outputs in 1M context
            — dispatches Handler calls (≤5 concurrent), judges each output against the
              lean-lesson-spec, decides done / retry / failed, updates the manifest
            — IS the two-tier quality gate (structure-complete + ship-ready), reasoning-judged
            — escalates to Conductor: writes BLOCKED to status.json when a batch can't resolve

            └─ GLM-5.1 / GLM-5.1v Handlers (content writers)
                 — 5.1: lessons, exercises, quizzes (text)
                 — 5.1v: illustrations, diagrams (vision)
                 — write to spec; a rejected output is re-prompted by the Taskmaster
                 — escalate: rejected 3x → Taskmaster marks the row failed

                 └─ Peons (GLM-5-Turbo — OPTIONAL, mostly absorbed)
                      — GLM-5.2's reasoning now handles most interrupt resolution inline
                      — keep only for cheapest high-volume synchronous lookups
```

**Key principle:** the GLM-5.2 Taskmaster is one large-context oversight process per batch cycle; the Handlers are ≤5 concurrent worker calls. The worker budget applies to Handlers, not to the Taskmaster.

---

## Model & Config Matrix

| Tier | Model string | Context | max_tokens (output) | Role |
|------|-------------|---------|---------------------|------|
| Taskmaster | `glm-5.2` (`glm-5.2[1m]` for max-context oversight) | up to 1,000,000 | up to 131,072 | Oversee, dispatch, judge, update manifest |
| Text Handler | `glm-5.1` | standard | **32,000** (was 8,000 — truncation fix) | Write lessons / exercises / quizzes |
| Vision Handler | `glm-5.1v` | standard | 32,000 | Illustrations, diagrams |
| Peon (optional) | `glm-5-turbo` | small | 1,000 | Cheap synchronous lookups |

**Endpoint (all tiers):** `https://api.z.ai/api/coding/paas/v4` — unify here. The touch-up dispatcher's old `open.bigmodel.cn` endpoint with no token cap is deprecated; repoint it.

**Every Handler call now logs `finish_reason`.** `finish_reason == "length"` means a genuine output-cap truncation (should never happen at 32K/131K); anything else that stops short is a timeout/disconnect. This turns "is it truncated?" from a content heuristic into an API fact.

---

## Two Squads in One Ecosystem

| Squad | Members | Model | Job | Manifest |
|-------|---------|-------|-----|----------|
| **The Loom** | GLM-5.2 Taskmaster + Lyra handlers (GLM-5.1/5.1v) | 5.2 + 5.1 | Weave lessons (generate) | `stages/02-.../manifest.json` |
| **The Tailors** | Echo · Newton · Turing · Shannon · Hinton | GLM-5.2 | Verify the AI seam, cross-reference sources | `stages/09-.../manifest.json` |

The Loom builds; the Tailors finish. They run on **separate manifests** and **chain sequentially** (Loom → ship-ready → Tailors → Loom correction) so they never compete for the 5-call ceiling. Tailors verify-and-flag; they do not rewrite. Full squad: `TAILORS.md`. Hypatia (GLM-5.1) tailors the GTM seam alongside.

---

## Existing Handlers

### Stage 02 — Lesson Injection Handler
**File:** `skills/operator-kit/dispatch-stage02.py`
**Reads:** `stages/02-lesson-injection/output/manifest.json`
**Writes:** `stages/02-lesson-injection/output/hybrid-lessons/<phase>/<lesson>/docs/en.md`
**Model:** `glm-5.1`, `max_tokens=32000`, `timeout=180`, `stream=True`, logs `finish_reason`
**Overseen by:** GLM-5.2 Taskmaster (judges output against `shared/lean-lesson-spec.md`)
**Recovery:** `run.ps1 stage02 --retry-failed`

### Stage 03 — Exercise Design Handler
**File:** `skills/operator-kit/dispatch-stage03.py`
**Model:** `glm-5.1`, `max_tokens=8000` (raised from 2000), logs `finish_reason`
**Recovery:** `run.ps1 stage03 --retry-failed`

### Stage 04 — Quiz Bank Handler
**File:** `skills/operator-kit/dispatch-stage04.py`
**Model:** `glm-5.1`, `max_tokens=8000` (raised from 2500 — the 40% JSON-parse failures were truncation), `response_format=json` where supported, logs `finish_reason`
**Output validation:** valid JSON, exactly 6 questions
**Recovery:** `run.ps1 stage04 --retry-failed`

### Watchdog — Stall Monitor
**File:** `skills/operator-kit/watchdog.py`
**Behavior:** polls status.json every 60s; stall = no manifest progress in 5 min; graceful stop → restart `--retry-failed`. Escalates to Conductor after 3 failed auto-recoveries.

---

## Taskmaster Folder Structure

```
skills/operator-kit/taskmasters/
  stage02-injector/  CONTEXT.md  status.json
  stage03-exercises/ CONTEXT.md  status.json
  stage04-quizzes/   CONTEXT.md  status.json
```

Each `CONTEXT.md` now briefs the **GLM-5.2 orchestrator** (what to run, the spec to judge against, where outputs go), not a Claude sub-agent. The launch trigger IS the instruction.

---

## Global Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| Max concurrent GLM **Handler** calls | 5 | Z.ai safe ceiling; 63% failure at 8 (observed) |
| Taskmaster oversight calls | 1 per batch cycle | Large-context, not per-item; outside the worker budget |
| API timeout per call | 180s | Reasoning models hang on dropped connections |
| Global backoff on 429 | 30s | All workers pause |
| Circuit breaker | 30% failure in 10 jobs → pause 60s | Catches sustained Z.ai degradation |
| Handler max_tokens | 32,000 | 131K headroom available; 32K is generous for a lean lesson |
| finish_reason logging | always | Truncation becomes an API fact, not a guess |

---

## Escalation Quick Reference

| Who sees the problem | Tells | How |
|---------------------|-------|-----|
| Handler output rejected | GLM-5.2 Taskmaster | re-prompt; 3x → mark row `failed` |
| Taskmaster can't resolve batch | Conductor | write BLOCKED to status.json |
| Conductor blocked | Director | one message: what's blocked, what was tried |
| Watchdog auto-recovery fails 3x | Conductor | watchdog-recovery.log entry |

---

## Token-Spend Split (target: 85% GLM / 15% Conductor)

| Work | Tier | Share |
|------|------|-------|
| Lesson/exercise/quiz generation | GLM-5.1 Handlers | majority |
| Batch oversight, quality judging, retry decisions | GLM-5.2 Taskmaster | the share that used to be Claude sub-agents |
| Architecture, gates, chaining, commits, Director comms | Conductor (Claude) | ~15% |

The Taskmaster migration is what moves the line to 85/15: oversight tokens shifted from Claude to GLM-5.2 (plan-covered, $0).
