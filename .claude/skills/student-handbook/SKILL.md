---
name: student-handbook
version: 1.0.0
description: >-
  Index of learner resources in this repo — skills, rules, progress files,
  modes, and tools. Use with /student-handbook, "student handbook", "what
  resources do I have", or when iterating how you study here.
disable-model-invocation: true
---

# Student handbook

You asked for a single map of **what exists and what it is for**. Present this structure; offer to go deeper on one row if the learner asks.

---

## Start here (learner path)

| Step | What | Invoke / open |
|------|------|----------------|
| 1 | Starting phase + path | README phase tables, `progress/learning-profile.json` |
| 2 | How you learn best (preferences, doubts) | `/learning-style-setup` |
| 3 | See all resources (this page) | `/student-handbook` |
| 4 | Study with progress + bridges | `.claude/rules/curriculum-chat` (automatic in Agent) |
| 5 | Open questions on curriculum | Office hours — say "office hours" or set `focus.mode` to `office-hours` in progress file |
| 6 | Doubts, motivation, preferences | `/guidance-counselor` |
| 7 | After finishing a phase | `/check-understanding <phase>` |
| 8 | Save trail across machines | Site **Save** → `progress/aifs-progress.json` (see `progress/README.md`) |

**Curriculum steps:** `README.md` lesson tables → `ROADMAP.md` shipping status (maintainers) → `phases/NN-phase/MM-lesson/docs/en.md`.

---

## .claude/skills/

| Skill | Purpose | When |
|-------|---------|------|
| **student-handbook** | This index | Lost, onboarding, iterating your setup |
| **learning-style-setup** | Preferences, strengths, weaknesses, doubts → `progress/learning-profile.json` | Once, then when something changes |
| **guidance-counselor** | Meta: doubt, pace, preference — not lesson drill | Stuck emotionally, "am I cut out for this", how to study |
| **check-understanding** | Phase exam (8 Q) | After a phase |
| **lesson-planning** | How quizzes should align with lessons; review quality | You or agents authoring/reviewing `quiz.json` — not daily study |
| **housekeeping** | Post-sprint bloat audit; report before any cleanup | `/housekeeping`, reduce bloat, consolidate rules/skills |
| **refresh** | Rebuild or check local code graph | `/refresh`, graph refresh, is the graph stale |
| **sub-agent-dispatch** | Middle-tier task delegation pattern for Claude Code | When Claude Code needs to investigate before deciding, but the work is below horizon level |

Lesson **outputs** under `phases/**/outputs/` (hundreds of prompts/skills) are lesson artifacts — install via `scripts/install_skills.py` or README skills.sh section.

---

## .claude/rules/

| Rule | Purpose |
|------|---------|
| **curriculum-chat** | Read `progress/aifs-progress.json`; office hours vs lesson quiz flow; cross-phase bridges |
| **lesson-planning-gate** | Before editing `quiz.json`, read lesson-planning skill |
| **local-graph.md** | Query-first, refresh via `/refresh`, verify gate |
| **subagent-first.md** | Delegate bulk/parallel work to subagents (Cline) |
| **terminal-hygiene** | Cline backend terminal vs editor terminals |

---

## Progress files (`progress/`)

| File | Purpose | Who writes |
|------|---------|------------|
| **aifs-progress.json** | Lessons visited, quiz answers, optional `focus` | Site Save → you commit; agents **read only** |
| **learning-profile.json** | Learning style, doubts, preferences | You save after `/learning-style-setup`; agents **read only** |
| **README.md** | Schema for both files | Reference |

`focus` example for agents:

```json
"focus": {
  "currentLesson": "phases/07-transformers-deep-dive/15-attention-variants",
  "mode": "office-hours",
  "note": "confused about SWA"
}
```

Modes: `office-hours` | `lesson` | `guidance` (see guidance-counselor skill).

---

## Agents (who does what)

| Who | Role | Use for |
|-----|------|---------|
| **Claude Code** | Dean · Horizon Coder | Lesson planning, curriculum architecture, new content, batch briefs |
| **Cline** | Professor · Inline Coder | Student teaching and tutoring, file edits, commits, quiz factory execution |

---

## quiz-factory/ (maintainers — not study)

Batch queue to finish missing `quiz.json` files. Uses **lesson-planning** skill + `manifest.json`. Learners ignore unless you are closing the curriculum gap.

---

## Site (learner)

| Piece | Purpose |
|-------|---------|
| [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com) | Catalog, lessons, quizzes, localStorage progress |
| **Save / Load** | Export `aifs-progress.json` for repo + agents |

Site does not write `learning-profile.json` or `focus` yet — edit in repo or ask Agent to draft JSON for you to save.

---

## Reference docs (repo root)

| Doc | Purpose |
|-----|---------|
| **AGENTS.md** | Lesson contract, quiz schema, CI, commit rules |
| **ROADMAP.md** | Which lessons exist / WIP (not your personal progress) |
| **glossary/terms.md** | Shared vocabulary |
| **GRAPH_REPORT.md** | Optional architecture skim (agents use `query_graph.py` first) |

---

## Modes (how to talk to Agent)

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Lesson** | Open lesson path, "do this lesson", `focus.mode: lesson` | Pre → teach → check → post (`quiz.json`) |
| **Office hours** | "office hours", broad topic | No forced lesson quiz; bridges from progress |
| **Guidance** | `/guidance-counselor` | Meta: doubts, preferences; reads `learning-profile.json` |
| **Placement** | README + `learning-profile.json` | Starting phase (placement skill archived) |
| **Phase review** | `/check-understanding N` | End-of-phase quiz |

---

## Planned / not built (iterate later)

Document here so expectations stay honest:

| Idea | Status | Notes |
|------|--------|-------|
| **Job counselor** | Not built | Future: aspirations, role fit, plan adjustments — needs separate setup + privacy thought |
| **Interview prep** | Not built | Infer: extend from capstone / agent phases when you add lessons |
| **Resume workshop** | Not built | Same — tie to phase 19 + job counselor later |
| **ROADMAP as career steps** | Partial | ROADMAP = curriculum shipping; career mapping = future job counselor |

When the learner asks for these, say they are on the handbook backlog; offer guidance-counselor for study habits and README phase tables for path.

---

## Maintainer vs learner

| You are… | Use |
|----------|-----|
| Learning | Skills table + progress files + site |
| Fixing curriculum | lesson-planning + quiz-factory + Cline |
| Housekeeping / bloat | `/housekeeping` (audit first, approve, then execute) |
| Confused about tooling | This handbook |

---

## Iterating this handbook

Edit `.claude/skills/student-handbook/SKILL.md` when you add skills, rules, or modes. Keep one row per resource; delete hype. Commit when ready.
