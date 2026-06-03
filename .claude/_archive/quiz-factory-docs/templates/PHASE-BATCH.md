# Phase BATCH — defaults

Copy to `phases/<this-phase-dir>/BATCH.md` and replace placeholders when starting this phase.

## Phase

- **Dir:** `phases/NN-phase-slug/`
- **Focus:** _one sentence — what learners should master in this phase_

## Scrape hints

- Primary doc sections: _e.g. "## Architecture", "## Build vs Use"_
- Code: _e.g. `code/main.py` only vs entire `code/` tree_
- Vocabulary: see `glossary/terms.md` for _term1, term2_

## Style anchor

- Best quiz in this phase (if any): `phases/NN-phase-slug/MM-lesson/quiz.json`
- Else use gold examples in `quiz-factory/REFERENCES.md`

## Common distractor patterns

- _e.g. confuse parameter count with FLOPs; confuse offline vs online RL_

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
