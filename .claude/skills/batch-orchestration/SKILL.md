---
name: batch-orchestration
description: >-
  Orchestrate large batch jobs using hierarchical subagents where each agent receives
  scoped knowledge (a context file), a narrow folder target (3–5 paths), and exactly
  one job type — implementing Anthropic's orchestrator-subagent workflow pattern through
  file architecture alone, without a runtime framework. Use when processing 50+ files
  across a corpus, curriculum, or codebase using parallel subagents. Trigger phrases:
  "batch subagents", "quiz factory", "folder-scoped agents", "parallel generation",
  "dispatch subagents", "hierarchical batch".
---

# Batch Orchestration via File Architecture

The core insight: **the file system is the tool**. You do not need a runtime framework
to implement Anthropic's orchestrator-subagent pattern. Give each subagent a scoped
knowledge file, a folder slice, and one job type — it becomes a focused agent with
bounded authority.

## The three ingredients per subagent

```
1. Scoped knowledge  →  BATCH.md (phase/domain context, style anchor, distractors)
2. Folder target     →  3–5 explicit file paths (never "all files in phase X")
3. Job type          →  one of: create_quiz | fill_explanations | schema_repair | redo_quiz
```

That combination replaces a wall of instructions. The subagent knows what to do, where
to do it, and what style to match — without needing the full project context.

## The hierarchy

```
Category (job type)
  └── Phase / Domain
        └── Batch  ← what each subagent receives (3–5 paths)
```

**Round** = up to 8 subagents in parallel. I launch, await, review, commit.
Subagents never touch git. One commit per output file (AGENTS.md hard rule).

## Queue: manifest.json

`quiz-factory/manifest.json` is the queue. Each row has:
- `path` — the target folder
- `phase` — two-digit phase number
- `job_type` — determines which BATCH.md and repair pattern to use
- `status` — `pending` | `done` | `blocked`
- `note` — repair instruction or block reason

Priority order: `redo_quiz` (0) → `schema_repair` (1) → `create_quiz` / `fill_explanations` (2).

Regenerate: `python3 quiz-factory/scripts/generate_manifest.py --write`

## Scoped knowledge: BATCH.md

Every phase that has pending work needs a `phases/NN-slug/BATCH.md`. It contains:
- **Focus**: one sentence on what learners master in this phase
- **Scrape hints**: which doc sections and code symbols to cite
- **Style anchor**: path to the best existing quiz in this phase (or a cross-phase gold)
- **Common distractor patterns**: what the doc explicitly refutes

If the file does not exist, use `quiz-factory/templates/PHASE-BATCH.md` defaults.

## Adding a new batch category

1. Define the detection logic in `generate_manifest.py` (pattern to detect, note to emit).
2. Add the repair/creation instructions to `quiz-factory/CLAUDE.md` under a new `## <job_type> job type` section.
3. Create `BATCH.md` for any phase that does not have one.
4. Run `generate_manifest.py --write` to emit the new rows.
5. Launch the first round.

## Orchestrator protocol (my role)

```
1. Pull next N batches from manifest (group by phase, 5 lessons/batch, 8 batches/round).
2. Launch all 8 subagents in parallel (Task tool, run_in_background=true).
3. Await completions (end turn; system notifies on finish).
4. For each completed batch:
   a. Verify stage sequence and correct indices with audit_lessons.py.
   b. Commit each lesson individually: git add phases/NN/MM && git commit -m "fix(...)".
   c. Mark manifest rows done/blocked.
5. Roll next round immediately.
```

## Subagent prompt template

```
You are a quiz factory batch worker for the curriculum at <repo root>.

Job type: <job_type>
Phase: <NN-phase-slug>

Lessons to process (all N):
1. phases/<phase>/NN-<lesson>
...

Phase context (BATCH.md):
<paste BATCH.md inline>

Repair/creation pattern:
<paste the relevant section from quiz-factory/CLAUDE.md>

Output: write quiz.json to each lesson directory. Report stage sequence,
correct indices, and DONE/BLOCKED for each lesson.
```

Nothing else in the prompt. No audit history. No rubric tables.

## Files in this repo

| File | Role |
|------|------|
| `quiz-factory/manifest.json` | The queue |
| `quiz-factory/scripts/generate_manifest.py` | Rebuilds the queue |
| `quiz-factory/CLAUDE.md` | All job-type instructions for subagents |
| `phases/NN-*/BATCH.md` | Per-phase scoped knowledge |
| `quiz-factory/QUALITY-STANDARDS.md` | 18-criterion scoring rubric |
| `quiz-factory/REFERENCES.md` | Schema, variance rule, anti-patterns |

## Connection to Anthropic workflow patterns

This is the **orchestrator-subagents** pattern (Lesson 14/12) implemented without
a framework:

- **Orchestrator** = me (parent agent): holds the queue, reviews outputs, commits
- **Subagents** = workers: each has a single tool (write to folder) and scoped knowledge
- **Parallelization** = 8 agents per round, all same job type, different folders
- **Knowledge injection** = BATCH.md replaces a system prompt or tool definition
- **Tool boundary** = the folder path IS the tool boundary; agents cannot drift outside it

The pattern scales to any corpus where work can be decomposed into: job type × domain × item slice.
