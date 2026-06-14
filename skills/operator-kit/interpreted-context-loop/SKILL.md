# /interpreted-context-loop (ICL)

**Interpreted-Context-Loop** — a recursive goal pattern where the loop's state lives in the environment, not the session. Any agent can cold-start the loop from the manifest alone and continue exactly where the previous agent left off.

---

## What ICL Is

A loop is not a running process. It is:

1. **A document describing a recursive goal** — what the loop is trying to accomplish, what done looks like, what triggers the next iteration
2. **A manifest tracking position in that goal** — structured state (not narrative) that any agent can read to know what's finished, what's pending, what failed
3. **An exit condition in the manifest itself** — when pending hits zero, the loop terminates. No session required to decide this.

The loop runs, updates the manifest, exits. Any trigger — a new session, a watchdog, a Taskmaster poll — points an agent at the same CONTEXT.md. The agent reads the manifest, continues from current state, exits again. The loop never needs to be "alive." It just needs to be readable.

---

## How ICL Differs from ICM

| | ICM | ICL |
|---|---|---|
| **Pattern** | Folder structure — context lives in the environment | Recursive goal — loop state lives in the manifest |
| **Scope** | How agents orient themselves | How goals are pursued across multiple agent invocations |
| **Artifact** | CONTEXT.md + status.json | manifest.json with status per work item |
| **Question it answers** | "What am I supposed to do?" | "Where are we in doing it?" |

ICL is built on top of ICM. The folder (ICM) gives the agent its situation. The manifest (ICL) gives the agent its position in the recursive goal. Together they make a loop that is both cold-startable and self-terminating.

---

## The Manifest as Loop Counter

Traditional loops use a counter or cursor in memory. ICL uses the manifest:

```
pending  → not yet processed (loop continues)
done     → successfully processed (loop counts progress)
failed   → needs retry or skip decision (loop branches)
finished → all pending exhausted (loop exits)
```

Any agent reading this knows instantly: is the loop still running? How far along? What needs attention? No conversation history needed. No STATE.md narrative to parse. Just structured state.

---

## This Implementation — BlueSkyGTM Pipeline

The Stage 02 dispatcher is a concrete ICL:

- **Recursive goal:** generate a lesson for every slot in the manifest
- **Manifest:** `stages/02-lesson-injection/output/manifest.json` — 510 rows, each with status
- **Loop iteration:** dispatcher picks next pending row, calls GLM-5.1, writes output, marks done
- **Cold start:** Taskmaster reads CONTEXT.md, launches dispatcher, polls manifest — works identically whether it's the first run or the fifth restart after a crash
- **Exit condition:** pending = 0 → dispatcher exits → Taskmaster reports to Conductor → Stage 03 chains

The loop ran for hours across multiple sessions, survived process kills, and completed without the original session staying alive. That's ICL working as designed.

---

## ICM + ICL — How They Chain

```
Director sets goal
  └─ Conductor reads HIERARCHY.md (ICM)          ← orients itself
       └─ Conductor spawns Taskmaster             ← points at CONTEXT.md
            └─ Taskmaster reads CONTEXT.md (ICM) ← orients itself  
                 └─ Taskmaster launches dispatcher
                      └─ Dispatcher runs ICL      ← reads manifest, iterates, exits
                           └─ Manifest = done
                                └─ Taskmaster reports
                                     └─ Conductor chains next stage (new ICL)
```

Each stage is its own ICL. Stages chain when the previous manifest closes. The Conductor is the only agent that holds the chain — and even that is documented in HIERARCHY.md, so a new Conductor can pick it up cold.

---

## Minimum Viable ICL for Any Domain

```
project/
  manifest.json          ← { id, status, input, output } per work item
  agents/
    <role>/
      CONTEXT.md         ← what the loop is doing, exit condition, chain target
      LOOP.md            ← the recursive goal definition (what "done" means globally)
  loop-runner.py         ← reads manifest, processes pending, writes status, exits
```

**LOOP.md template:**
```markdown
# Loop: <Name>

## Goal
<What this loop accomplishes when all rows are done>

## Iteration
1. Read manifest — find next pending row
2. Process it — <what processing means here>
3. Write output — <where>
4. Mark done/failed in manifest
5. Repeat until pending = 0

## Exit Condition
pending = 0 → write finished: true → exit

## Chain Target
On exit, trigger: <next loop or human gate>

## Failure Policy
failed > 10%: pause and report
failed rows: retry once, then mark skip
```

---

## When to Invoke This Skill

- Designing any multi-step process that needs to survive session boundaries
- Debugging a stalled loop (check manifest status distribution first)
- Chaining stages — ICL exits trigger the next ICL
- Applying the pattern to a new domain (sales pipeline, content ops, data processing)
- Explaining why this architecture doesn't need persistent scheduling infrastructure

---

## Related

- [`/interpreted-context-manifest`](../interpreted-context-manifest/SKILL.md) — the folder pattern ICL runs inside
- [`HIERARCHY.md`](../HIERARCHY.md) — pipeline entry point
- [`HANDLERS.md`](../HANDLERS.md) — agent registry
- [`TASKMASTER-PROTOCOL.md`](../TASKMASTER-PROTOCOL.md) — how Conductor spawns loop runners
