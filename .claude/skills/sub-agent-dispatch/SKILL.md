---
name: sub-agent-dispatch
version: 1.1.0
description: >-
  When and how Claude Code (Dean) deploys sub-agents for middle-tier investigation —
  read-only research tasks that inform a decision or Cline brief. Not for execution.
  Execution of non-horizon batch work uses batch-orchestration instead.
disable-model-invocation: true
---

# Sub-agent dispatch

## The three-tier model

| Tier | Channel | Examples |
|------|---------|---------|
| **Horizon** | Claude Code acts directly | Lesson design, curriculum architecture, batch brief authorship, role definitions, infrastructure decisions |
| **Middle — investigate** | Sub-agent → reports to Claude Code | Audits, file surveys, targeted searches, state verification before a decision |
| **Middle — execute** | Batch-orchestration skill | Quiz factory batches, multi-file edits, parallel write jobs across phases |
| **Inline** | Cline (via user) | File edits, commits, student teaching |

Sub-agents are **read-only investigators**. They find and report. They never write files.  
Batch-orchestration handles execution that isn't horizon and isn't single-file inline.

---

## When to dispatch a sub-agent

Dispatch when ALL of these are true:
- You need information from files you don't already have in context
- The answer requires searching across multiple paths
- The output is a **report** — not a file change
- You would otherwise read 3+ files yourself just to make a decision

**Do NOT dispatch when:**
- You already have the relevant context in this conversation
- A single Read or Grep would answer the question
- The task involves writing, editing, or committing anything — that's Cline or batch-orchestration

---

## Dispatch as many as possible in parallel

In a single message, launch all independent sub-agents at once. Each gets a narrow, non-overlapping question. Do not sequence agents that don't depend on each other.

Example from this session (3 parallel):
- Agent 1: read batch-orchestration skill
- Agent 2: assess `outputs/` directory and external references to it
- Agent 3: read quiz-factory/REFERENCES.md

All three launched in one message, all returned independently, all synthesized together.

---

## How to brief a sub-agent

Give it:
1. **One focused question** — not "explore the repo"
2. **Starting paths** — where to look
3. **Exact output format** — "list file paths", "quote section X", "answer yes/no for each"
4. **A word/item cap** — keep the return small

Template:
```
Search focus: <one sentence>
Starting paths: <comma-list>
Return: <list / quote / count / boolean>
Cap: <word or item limit>
```

---

## What to do with the report

Sub-agents return findings, not decisions. Claude Code:
- **Synthesizes** the report into a clear decision
- **Writes a Cline brief** if file changes are needed (drops to `cline-backlog/batches/pending/`)
- **Launches batch-orchestration** if it's a parallel execution task across many files
- **Acts directly** if only horizon-level reasoning was needed

Sub-agent output goes to the user only as a synthesized Dean summary — not raw dump.

---

## Relationship to batch-orchestration

| Sub-agent dispatch | Batch-orchestration |
|--------------------|---------------------|
| Read-only | Read + write |
| One round, reports back | Multi-round, parallel workers |
| Used before deciding | Used to execute after deciding |
| Explore subagent type | General or task subagent |
| Result: Dean makes a call | Result: quiz.json files written and committed |

When a job needs both: dispatch first to assess scope, then use batch-orchestration to execute.

---

## Related

- Execution of non-horizon batch work: `.claude/skills/batch-orchestration/SKILL.md`
- Cline inline briefs: `cline-backlog/batches/pending/`
- Repo search tools: `scripts/query_graph.py`, `scripts/audit_lessons.py`
