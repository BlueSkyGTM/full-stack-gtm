# /interpreted-context-manifest (ICM)

**Interpreted Context Methodology** — agents understand their situation by reading the environment around them, not by receiving verbal instructions. The manifest is what makes that environment legible.

---

## What ICM Is

ICM is a design pattern for multi-agent systems where:

1. **Context lives in folders, not conversations.** Every agent gets a folder. The folder contains everything it needs to orient itself: what it owns, what it reads, what it writes, how to escalate.
2. **The manifest is the map.** A structured state file (manifest.json, status.json, or equivalent) turns a folder into a situation the agent can interpret — not just files, but a state machine with positions, statuses, and destinations.
3. **The upstream trigger is the instruction.** When an agent is handed a path to its folder, that handoff IS the instruction. No verbal briefing. No repo dump. The agent reads its surroundings and derives what to do.
4. **Agents are re-entrant by design.** Because context lives in persistent files (not in session memory), any agent can be killed, restarted, or replaced and pick up exactly where the previous one left off.

---

## This Implementation — BlueSkyGTM Curriculum Pipeline

This codebase implements ICM around a content generation pipeline. The specific artifacts:

| Artifact | Role in ICM |
|----------|-------------|
| `stages/0X-*/output/manifest.json` | The manifest — 510 slots, each with phase, lesson, status, output path |
| `skills/operator-kit/taskmasters/*/CONTEXT.md` | Agent folders — each Taskmaster reads this cold and knows exactly what to run |
| `skills/operator-kit/HIERARCHY.md` | Entry point for any new agent joining the system |
| `skills/operator-kit/HANDLERS.md` | Full registry of tiers, models, failure modes, escalation |
| `skills/operator-kit/TASKMASTER-PROTOCOL.md` | How Conductor spawns Taskmasters — the thin launcher pattern |

**Five-tier agent stack:**
```
Director (human)
  └─ Conductor (Claude Code)
       └─ Taskmasters (Claude sub-agents — thin launchers)
            └─ Handlers (GLM-5.1 — content writers)
                 └─ Peons (GLM-5-Turbo — synchronous interrupt resolvers)
```

**How to spawn a Taskmaster (Conductor does this):**
```
Point Agent tool at: skills/operator-kit/taskmasters/<role>/CONTEXT.md
The sub-agent reads it. That's the full instruction.
```

**How to check pipeline state:**
```powershell
python -c "
import json
with open('stages/02-lesson-injection/output/manifest.json', encoding='utf-8') as f:
    rows = json.load(f)
s = {}
for r in rows: s[r['status']] = s.get(r['status'], 0) + 1
print(s)
"
```

---

## Applying ICM Elsewhere

ICM is not specific to curriculum pipelines. The pattern applies anywhere agents need to be:
- **Fault-tolerant** — survives session boundaries, process kills, partial failures
- **Re-entrant** — a replacement agent can pick up mid-task without a handoff conversation
- **Parallelizable** — multiple agents can work the same manifest without coordination overhead

**The minimum viable ICM implementation for any domain:**

```
<project>/
  agents/
    <role-name>/
      CONTEXT.md      ← what this agent owns, reads, writes, escalates
      status.json     ← current state (written by the running process)
  manifest.json       ← the map: every work item with id, status, input, output
```

**CONTEXT.md template:**
```markdown
# Agent: <Role Name>

## What You Own
- Stage/scope: ...
- Input: ...
- Output folder: ...
- Status file: ...

## What You Do
1. Read manifest — count pending/failed items
2. Launch subprocess / call tool
3. Poll status file every 60s (read only: done, failed, pending, finished)
4. Report to upstream when finished

## What to Report
- Total done / failed / pending
- Any items with >3 failures
- Whether next stage can proceed

## Rules
- Do not read output files into your context
- Do not guess — report BLOCKED if uncertain
- Escalation: subprocess non-zero exit → retry once → report BLOCKED
```

---

## When to Invoke This Skill

- Designing a new multi-agent system from scratch
- Onboarding a new agent to the pipeline cold
- Debugging why an agent is stuck (check its CONTEXT.md + status.json)
- Extending the pipeline to a new stage (create a new taskmasters/<role>/ folder)
- Applying ICM to a domain outside this codebase

---

## Related Files

- [`HIERARCHY.md`](../HIERARCHY.md) — pipeline entry point, quick start
- [`HANDLERS.md`](../HANDLERS.md) — full agent registry
- [`TASKMASTER-PROTOCOL.md`](../TASKMASTER-PROTOCOL.md) — thin launcher pattern
- [`vault/plan-agent-hierarchy.md`](../../../vault/plan-agent-hierarchy.md) — design decisions and audit trail
