---
name: lesson-planning
version: 1.0.0
description: >-
  Lesson and quiz planning for AI Engineering from Scratch — how assessment
  aligns with docs and code. Use for /lesson-planning, reviewing quiz-factory
  output, auditing weak lessons, or planning new lesson content and quizzes.
disable-model-invocation: true
---

# Lesson planning

Teacher-style reference: how a lesson unit fits together and how quizzes should attach to material that already exists. Use when reviewing Claude Code batch work, spot-checking `quiz.json`, fixing weak lessons, or drafting new content.

**Required by rule:** `.claude/rules/lesson-planning-gate.md` — any agent creating or editing `quiz.json` (or advising on quiz quality) must read this skill first.

**Machine batching** (manifest, commits, audit commands) stays in `quiz-factory/`. This skill is for **judgment and comparison**.

---

## When to use

- `/lesson-planning` or "review this quiz", "plan quiz for lesson X"
- Comparing a new or factory-generated `quiz.json` to the bar set by flagship lessons
- A lesson feels **crappy** — decide whether the problem is `docs/en.md`, `code/`, quiz, or catalog
- **New content** — outline objectives before writing quiz or code
- PM spot-check after a `quiz-factory` phase slice

---

## Source of truth (order)

1. `phases/.../docs/en.md` — objectives, concept depth, prerequisites (difficulty intent)
2. `phases/.../code/` — what Build lessons actually run
3. `AGENTS.md` — `quiz.json` schema (6 questions, stages, Tier A/B)
4. `quiz.json` — downstream; must not drive rewrites of the lesson doc

---

## Lesson unit map

| Piece | Role |
|-------|------|
| README / ROADMAP row | Learner can find the lesson |
| `docs/en.md` | Teach: Problem → Concept → Build/Use |
| `code/` | Prove (Build) or support (Learn) |
| `quiz.json` | Assess: pre → teach → check×3 → post×2 on site |

Quiz difficulty = "understood **this** lesson," not generic ML trivia.

---

## Stage expectations

| Stage | Learner should… | Pull from doc… |
|-------|-----------------|----------------|
| pre | Recall the pain or headline tradeoff | The Problem, hook |
| check (×3) | Apply three **different** mechanisms | Concept, tables, equations |
| post (×2) | Integrate two ideas or doc + code | Table + prereq named in doc, or asymptotics + demo |

**Distractors:** wrong options the **lesson refutes**, not silly fillers.  
**Explanations:** 1–3 sentences in lesson vocabulary; empty = incomplete (Tier B).

---

## Seven insights (flagship bar)

Read the live `quiz.json` when comparing. These are patterns that worked when closing orphan lessons — not text to copy.

### 1. Bridge prerequisites by name only when fair

**Lesson:** `phases/07-transformers-deep-dive/15-attention-variants/`

**Insight:** One check may contrast this lesson with a **listed** prerequisite (e.g. Lesson 12 FlashAttention = IO/kernel vs SWA = mask topology). Do not test off-syllabus lessons.

**Compare:** Does each cross-lesson mention appear in `docs/en.md` prerequisites or body?

---

### 2. Separate mechanism from production pattern

**Lesson:** `phases/07-transformers-deep-dive/15-attention-variants/`

**Insight:** SWA definition (pre/check) vs **why Gemma interleaves global layers** (check) vs differential sink (check) are three distinct claims — not one fact reworded.

**Compare:** Are your three checks non-overlapping?

---

### 3. Put the lesson’s math on a check when the doc owns it

**Lesson:** `phases/07-transformers-deep-dive/16-speculative-decoding/`

**Insight:** Acceptance uses min(1, q/p); pre states **distribution match** with verifier. Checks discriminate myths ("draft replaces verifier", "verifier never runs").

**Compare:** If the doc has a formula or rule, is there exactly one check that tests it as stated?

---

### 4. Paradigm shift beats vocabulary trivia

**Lesson:** `phases/08-generative-ai/19-visual-autoregressive-var/`

**Insight:** Questions target **next-scale vs pixel order** and multi-scale VQ — not ImageNet scores. Distractors are wrong **generation orders** the doc rejects.

**Compare:** Generative lessons — are distractors alternate paradigms, not unrelated metrics?

---

### 5. Same topic later in the path ≠ same quiz

**Lessons:** `07/16-speculative-decoding` vs `10/15-speculative-decoding-eagle3`

**Insight:** Phase 10 lesson must scrape **EAGLE3 doc claims**, not reuse 07/16 JSON. Same family, different objectives and Build demo.

**Compare:** Sequel or phase-advanced lesson — quiz on **delta**, not full re-teach?

---

### 6. Variant "v2" lessons test what changed

**Lesson:** `phases/10-llms-from-scratch/16-differential-attention-v2/`

**Insight:** Objectives drive checks on **v2 vs baseline**, not re-defining softmax. Pair with 07/15 only if prerequisites say so.

**Compare:** Is at least one check clearly "what's new here"?

---

### 7. Build lessons owe one post to the demo

**Lessons:** `07/15` (mask comparator), `10/34-gradient-checkpointing` (checkpoint tradeoff)

**Insight:** At least one check or post names what `code/main.*` shows — masks, acceptance loop, memory/FLOPs story. pre can be conceptual; post ties **running** the lesson.

**Compare:** Type: Build in frontmatter → is code referenced in post or a late check?

---

## Planning workflow (new or missing quiz)

1. Read objectives + Problem + Concept (+ Build It).
2. List 6–10 **claims** the doc asserts (testable sentences).
3. Assign claims to pre / check×3 / post×2 (injective — no reuse).
4. Draft four options per claim; index `correct`.
5. Write explanations from the winning claim only.
6. Run `python scripts/audit_lessons.py` (and `--strict-quiz` before fork-complete).
7. One commit per lesson dir: `feat(phase-NN/MM): add <slug> quiz`.

---

## Reviewing factory or Claude output

| Question | Pass |
|----------|------|
| Can you point to a paragraph in `docs/en.md` for each question? | required |
| Three checks test three different ideas? | required |
| Build lesson references code in check or post? | required |
| Distractors plausible but wrong per doc? | required |
| Explanations non-empty and teach? | required for Tier B |
| Copied wording from a flagship lesson on a **different** topic? | fail |

Flagship JSON paths for side-by-side read:

```text
phases/07-transformers-deep-dive/15-attention-variants/quiz.json
phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json
phases/08-generative-ai/19-visual-autoregressive-var/quiz.json
phases/10-llms-from-scratch/15-speculative-decoding-eagle3/quiz.json
phases/10-llms-from-scratch/16-differential-attention-v2/quiz.json
phases/10-llms-from-scratch/17-native-sparse-attention/quiz.json
phases/10-llms-from-scratch/34-gradient-checkpointing/quiz.json
```

---

## When a lesson is crappy (triage)

| Symptom | Likely fix |
|---------|------------|
| Quiz easy but doc is deep | Rewrite quiz from objectives; don't dumb down doc |
| Quiz hard but doc shallow | Fix `docs/en.md` first, then quiz |
| Quiz contradicts doc | Quiz wrong — fix quiz only |
| No quiz, good doc/code | Factory `create_quiz` or manual plan above |
| Quiz exists, empty explanations | `fill_explanations` job; read doc for feedback text |
| Lesson not in README | Catalog row, not quiz |

---

## New lesson content (order)

1. `docs/en.md` frontmatter + objectives (AGENTS.md contract)
2. `code/` if Build
3. README / ROADMAP link
4. `quiz.json` last — plan with this skill, then implement or queue in `quiz-factory/manifest.json`

---

## Related

- Batch operator: `quiz-factory/CLAUDE.md`, `quiz-factory/ARCHITECTURE.md`
- Schema + audit: `AGENTS.md`, `scripts/audit_lessons.py`
- Phase spot-check rubric: `quiz-factory/QUALITY-RUBRIC.md`
