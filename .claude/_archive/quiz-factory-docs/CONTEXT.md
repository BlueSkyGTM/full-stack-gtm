# Context (frozen)

## Product

- Repo: **AI Engineering from Scratch** — 472 lessons under `phases/`.
- This fork will **not** take upstream curriculum updates after cementing.
- Learners use `quiz.json` on the site: **1 pre, 3 check, 2 post** (see `AGENTS.md`).

## What is already done (do not redo)

- Catalog orphans wired; duplicate `10/25-speculative-decoding` removed.
- All **55** capstone quizzes schema-correct with real questions (incl. 54–57).
- Seven flagship lessons have new quizzes (07/15, 07/16, 08/19, 10/15-eagle3, 10/16, 10/17, 10/34).
- Phases **00–05, 11, 14, 19** have a `quiz.json` on every lesson (Tier A).

## What this factory finishes

| `job_type` | Meaning |
|------------|---------|
| `create_quiz` | No `quiz.json` — write full file (6 questions + explanations). |
| `fill_explanations` | `quiz.json` exists but one or more `explanation` fields empty. |

Regenerate queue: `python quiz-factory/scripts/generate_manifest.py`

## Default phase order (manifest `phase_order`)

Process pending rows in this order (skip phases with no pending rows):

1. `07` — finish transformers gap (13 lessons)
2. `08` — generative gap (14)
3. `10` — LLMs scratch gap (8)
4. `06` — speech (17)
5. `09` — RL (12)
6. `12` — multimodal (25)
7. `13` — tools/protocols (23)
8. `15` — autonomous (22)
9. `16` — multi-agent (23)
10. `17` — fill explanations (28 lessons, files exist)
11. `18` — fill explanations (30 lessons, files exist)

Phase **11** has four lessons with partial empty explanations — included in manifest when present.

## Source of truth for question difficulty

1. `phases/.../docs/en.md` — especially **Learning Objectives**
2. `phases/.../code/` — what Build lessons actually implement
3. Same-phase gold quiz in REFERENCES.md — style only, not content

Do not use the open web or generic ML trivia.

## Quality tiers

- **Tier A** (required every row): `python scripts/audit_lessons.py` exits 0 for the repo.
- **Tier B** (target before fork complete): `python scripts/audit_lessons.py --strict-quiz` — run after each phase slice or at end.

`AGENTS.md` says "do not bulk-generate quizzes" for ad-hoc agents. **This directory is the sanctioned bulk path.**
