# Quiz factory — batch operator (Claude Code)

You are a **batch worker**, not a tutor. Do not chat, ask questions, or redesign lessons.

**Gate:** `.cursor/rules/lesson-planning-gate.mdc` applies to every row you process.

## Mission

Process `manifest.json` rows with `status: pending` in phase order. For each row:

1. Follow [lesson-planning](../.cursor/skills/lesson-planning/SKILL.md) (alignment, difficulty, seven insights).
2. Run loops in [ARCHITECTURE.md](ARCHITECTURE.md).
3. Pass Tier A audit for that lesson.
4. Commit **one lesson directory only**.
5. Set manifest row to `done` (or `blocked` with reason).

## Hard rules

- **Read first:** [lesson-planning SKILL](../.cursor/skills/lesson-planning/SKILL.md) → [CONTEXT.md](CONTEXT.md) → [REFERENCES.md](REFERENCES.md) → [ARCHITECTURE.md](ARCHITECTURE.md).
- **Before each new phase:** read `phases/NN-*/BATCH.md`; use lesson-planning **seven insights** to compare output.
- **Do not** copy flagship `quiz.json` text — extract claims from **that** lesson’s doc/code (see skill).
- **Variance rule (mold):** the six `correct` values must not be a constant column; vary the correct slot (REFERENCES.md "Variance rule"). Leave position-dependent options ("all of the above", "both B and C") in place.
- **`redo_quiz` rows:** rebuild questions from the doc/code — the existing text is an anti-pattern (all answer `A`), do not trust it.
- **`schema_repair` rows:** follow the repair pattern in the row's `note` field (see `## schema_repair job type` below). Preserve every question that already meets quality standards; only add, drop, or strengthen as the note specifies.
- **Edit only** `phases/.../MM-lesson/quiz.json` unless `job_type` says otherwise.
- **Never** edit `docs/en.md`, `code/`, `README.md`, or `site/data.js`.
- **Never** batch multiple lessons in one commit.
- **Never** skip audit because the quiz "looks fine."
- On audit fail: fix quiz, re-audit, max **2** retries → then `blocked` and continue.
- Append one line per lesson to `run.log`: `ISO8601 | path | done|blocked | commit or error`.

## Phase layers

Before processing phase `NN`, read:

`phases/NN-*-slug/BATCH.md` if it exists, else `templates/PHASE-BATCH.md` (defaults only).

## Commit format

```text
feat(phase-NN/MM): add <lesson-slug> quiz
```

For explanation-only jobs:

```text
fix(phase-NN/MM): fill quiz explanations
```

## Stop conditions

- All rows for the assigned phase slice are `done` or `blocked`.
- User message says stop.
- Do not stop early because a row is tedious.

## schema_repair job type

The `note` field of each `schema_repair` row contains the exact repair pattern. Three patterns exist in this repo:

**Phase 04 — add checks (5q → 6q)**
Current stages: `pre, pre, post, post, post`
Target stages:  `pre, check, check, check, post, post`
Steps:
1. Keep the stronger of the two `pre` questions; drop the other.
2. Write **3 new `check` questions** from three distinct sections of `docs/en.md` — one per concept cluster. At least one check must require applying a formula or algorithm step, not just naming a term.
3. Keep the two strongest `post` questions; drop the third. If neither post integrates two doc ideas, rewrite one to do so.
4. For Build lessons: at least one check or post must name a specific function from `code/main.*`.

**Phase 05 — trim (8q → 6q)**
Current stages: `pre, pre, check, check, check, post, post, post`
Target stages:  `pre, check, check, check, post, post`
Steps:
1. Keep the stronger of the two `pre` questions; drop the other.
2. Keep all three `check` questions unchanged.
3. Keep the two strongest `post` questions; drop the third OR merge the weakest two into a single integration question requiring two doc ideas.
4. For Build lessons: verify at least one check or post names a function from `code/main.*`; add if missing.

**Phase 14 — trim + code-ref (7q → 6q)**
Current stages: `pre, pre, check, check, check, post, post`
Target stages:  `pre, check, check, check, post, post`
Steps:
1. Keep the stronger of the two `pre` questions; drop the other.
2. Keep all three `check` questions and both `post` questions unless they fail quality criteria.
3. Verify at least one check or post names a specific function from `code/main.*`; rewrite one question to add this if missing.

**Commit format for schema_repair:**
```text
fix(phase-NN/MM): repair quiz schema
```

## When blocked

Write `blocked` + short reason in manifest. Do not invent quiz content from general knowledge — only from that lesson's `docs/en.md` and `code/`.
