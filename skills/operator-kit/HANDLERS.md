# Handler Registry — Chain of Command

Every agent in this system is documented here. If you're an agent reading this file, you are in the right place. Find your tier, understand your scope, and follow your escalation path.

---

## The Hierarchy

```
Director (User)
  — intent, quality gates, stage advancement only
  — never types terminal commands

  └─ Conductor (Claude Code)
       — reads Director intent, plans execution, spawns Taskmasters
       — owns: vault/plan-*.md, run.ps1, this file
       — escalates to Director: stage advancement, billing decisions, quality failures

       └─ Taskmasters (Claude sub-agents via Agent tool)
            — thin launchers: read CONTEXT.md folder, spawn detached Python subprocess
            — monitor: status.json polling only (never accumulate lesson content in context)
            — exit when subprocess completes; report summary to Conductor
            — escalate to Conductor: subprocess fails 3x, scope drift detected

            └─ Handlers (GLM-5.1 — primary content writers)
                 — write curriculum content: lessons, exercises, quizzes
                 — delegate menial interrupts to Peons via dispatch-peon.py
                 — continue writing after Peon resolves the interrupt
                 — escalate to Taskmaster: output rejected 3x, ambiguous instructions

                 └─ Peons (GLM-5-Turbo / GLM-Flash — interrupt resolvers)
                      — resolve a specific blocking subtask and return
                      — called synchronously by Handlers mid-work
                      — exit immediately after returning result
                      — never hold state; never run in background
```

**Key principle:** Agents are invoked in stages, not all active simultaneously. A Peon call happens inside a Handler call. A Taskmaster call scopes one batch of Handler calls. Worker budget (max 5 concurrent GLM API calls) is enforced by the Python dispatcher, not by the agents themselves.

---

## Existing Handlers

### Stage 02 — Lesson Injection Handler
**File:** `skills/operator-kit/dispatch-stage02.py`  
**Invoked by:** Taskmaster or Conductor directly via `run.ps1 stage02`  
**Reads:** `stages/02-lesson-injection/output/manifest.json` (510 rows)  
**Writes:** `stages/02-lesson-injection/output/hybrid-lessons/<phase>/<lesson>/docs/en.md`  
**Model:** GLM-5.1 (`max_tokens=8000`, `timeout=120`, `stream=True`)  
**Brief source:** `stages/00-c-agent-setup/output/agent-briefs/lyra-content-brief.md` (governed maze — ≤500 tokens extracted)  
**Worker default:** 3 | **Safe max:** 5 | **Hard limit:** 6  
**Failure modes:**
- Output < 500 chars → retry (up to 3x) → mark `failed`
- 429 → global_rate_pause(30) → all workers pause → exponential backoff
- Circuit breaker → >30% failure in 10 jobs → global_rate_pause(60)
- Timeout → 120s hard limit → mark `failed`
**Recovery:** `run.ps1 stage02 --retry-failed`

---

### Stage 03 — Exercise Design Handler
**File:** `skills/operator-kit/dispatch-stage03.py`  
**Invoked by:** Taskmaster or Conductor via `run.ps1 stage03`  
**Reads:** Stage 02 manifest (done rows) + `hybrid-lessons/.../docs/en.md` per lesson  
**Writes:** `stages/03-exercise-design/output/exercise-specs/<phase>/<lesson>/exercises.md`  
**Model:** GLM-5.1 (`max_tokens=2000`, `timeout=120`, `stream=True`)  
**Worker default:** 3 | **Safe max:** 5  
**Failure modes:** Same as Stage 02. Strips `## GTM Redirect Rules` bleed automatically.  
**Recovery:** `run.ps1 stage03 --retry-failed`

---

### Stage 04 — Quiz Bank Handler
**File:** `skills/operator-kit/dispatch-stage04.py`  
**Invoked by:** Taskmaster or Conductor via `run.ps1 stage04`  
**Reads:** Stage 02 manifest (done rows) + Stage 03 exercise specs  
**Writes:** `stages/04-quiz-recall/output/quiz-bank/<phase>/<lesson>/cards.json`  
**Model:** GLM-5.1 (`max_tokens=2500`, `timeout=120`, `stream=True`)  
**Output validation:** Must be valid JSON with exactly 6 questions. Markdown fences stripped automatically.  
**Worker default:** 3 | **Safe max:** 5  
**Recovery:** `run.ps1 stage04 --retry-failed`

---

## New Handlers (built in this session)

### Watchdog — Stall Monitor
**File:** `skills/operator-kit/watchdog.py`  
**Invoked by:** Conductor (not Director)  
**Reads:** `stages/<N>-*/output/status.json` (any active stage)  
**Writes:** `skills/operator-kit/watchdog-recovery.log`  
**Behavior:** Polls every 60s. Detects stall (no progress in 5min or absent status.json after 2min). Graceful stop via PAUSE_SENTINEL before kill. Restarts with `--retry-failed`.  
**Escalates to Conductor:** When auto-recovery fails 3x in a row.

### Peon Dispatcher
**File:** `skills/operator-kit/dispatch-peon.py`  
**Invoked by:** GLM-5.1 Handlers mid-generation (synchronous call)  
**Task types:** citation lookup, format validation, metadata generation, text extraction, diagram generation  
**Models:** GLM-5-Turbo (all peon tasks including diagram SVG generation) · GLM-5.1 (Handlers — content writing)  
**Returns:** Validated result string or raises PeonError after 2 retries  
**Escalation:** PeonError returned to Handler → Handler marks subtask [CITATION NEEDED], [METADATA PENDING], or [DIAGRAM PENDING] and continues

---

## Taskmaster Folder Structure

Each Taskmaster role has its own folder under `skills/operator-kit/taskmasters/`:

```
skills/operator-kit/taskmasters/
  stage02-injector/
    CONTEXT.md          — Taskmaster brief: what to run, where outputs go
    status.json         — written on launch, updated by subprocess
  stage03-exercises/
    CONTEXT.md
    status.json
  stage04-quizzes/
    CONTEXT.md
    status.json
```

The Conductor reads `CONTEXT.md` to brief the sub-agent. The sub-agent reads `CONTEXT.md` on launch and knows everything it needs. The trigger from the Conductor IS the instruction.

---

## Global Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| Max concurrent GLM calls | 5 | Z.ai safe ceiling; 63% failure at 8 workers (observed) |
| API timeout per call | 120s | GLM-5.1 reasoning model can hang on dropped connections |
| Global backoff on 429 | 30s | All workers pause, not just the hit thread |
| Circuit breaker threshold | 30% failure in 10 jobs | Catches sustained Z.ai degradation |
| Circuit breaker pause | 60s | All workers via global_rate_pause(60) |
| Output min length | 500 chars (lessons) | Rejects thinking-only GLM output |
| Peon max retries | 2 | Handler continues with fallback on PeonError |

---

## Escalation Quick Reference

| Who sees the problem | Who to tell | How |
|---------------------|-------------|-----|
| Peon fails 2x | Handler | raise PeonError (caught in Handler) |
| Handler fails 3x | Taskmaster | mark row `failed`; Taskmaster sees it in status.json |
| Taskmaster subprocess fails 3x | Conductor | Taskmaster reports in Agent tool result |
| Conductor blocked | Director | Single message: what's blocked, what was tried |
| Watchdog auto-recovery fails 3x | Conductor | watchdog-recovery.log entry + Conductor polls it |
