# References

## Schema (canonical)

From `AGENTS.md` — every `quiz.json`:

```json
{
  "lesson": "<dir-slug>",
  "title": "<Lesson Title>",
  "questions": [
    {"stage": "pre", "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": "..."},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": "..."},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 2, "explanation": "..."},
    {"stage": "check", "question": "...", "options": ["a","b","c","d"], "correct": 1, "explanation": "..."},
    {"stage": "post", "question": "...", "options": ["a","b","c","d"], "correct": 3, "explanation": "..."},
    {"stage": "post", "question": "...", "options": ["a","b","c","d"], "correct": 0, "explanation": "..."}
  ]
}
```

- `correct`: zero-based index into `options`
- 2–6 options per question
- No legacy keys: `q`, `choices`, `answer`
- No substring `placeholder` in option text (Tier B / `--strict-quiz`)

## Variance rule (the mold) — non-negotiable

The schema above is the **mold**: it shows `correct` landing on `0,1,2,1,3,0` — never a constant column. Every quiz you create must obey:

1. **No constant answer key.** The six `correct` values must not all be the same index. A quiz where every answer is `A` (or every answer is `B`) is rejected — this is the exact defect found in the seven legacy quizzes (see Anti-pattern below).
2. **Spread the correct slot.** Write the correct option into a varied position — do not habitually place it first. Across the six questions, aim to touch at least three different indices.
3. **Mechanical, content-first.** Decide which option is *true* from `docs/en.md` first, then choose its slot. Never change a fact to hit a position.
4. **Position-dependent options stay put.** If an option says "all of the above", "both B and C", etc., leave that question's order alone — reordering would break the reference.

A repo-wide one-time fix already ran (`quiz-factory/scripts/balance_answers.py`). New quizzes must arrive varied so that tool never needs to touch them.

## Difficulty bar (one paragraph)

- **pre**: recall from hook + objectives (one step).
- **check**: apply the lesson's main mechanism; distractors = plausible mistakes from `docs/en.md`.
- **post**: integrate two ideas from the lesson or compare to prerequisite concept.
- **explanation**: 1–3 sentences; cite the lesson's terms; say why wrong options fail.

## Pedagogy (read before first draft)

| Doc | Content |
|-----|---------|
| [lesson-planning SKILL](../.claude/skills/lesson-planning/SKILL.md) | Seven insights, planning, review, triage |
| [QUALITY-RUBRIC.md](QUALITY-RUBRIC.md) | PM spot-check after each phase |

## Gold JSON (read live — insights in skill)

See **Seven insights** in lesson-planning for paths. Also: `phases/04-computer-vision/04-image-classification/quiz.json`, `phases/19-capstone-projects/54-paper-writer/quiz.json`.

## Scrape order (per lesson)

1. `docs/en.md` — title, objectives, key sections
2. `code/main.*` or primary source — function names, invariants, outputs
3. `outputs/*.md` — optional tone; do not copy blindly
4. Neighbor quiz in same phase if present

## Validation commands

From repo root:

```bash
python scripts/audit_lessons.py
python scripts/audit_lessons.py --strict-quiz
```

After one lesson (optional narrow check — audit still scans all lessons):

```bash
python scripts/audit_lessons.py
```

## Title field

Copy lesson title from `docs/en.md` H1 or frontmatter; match README lesson name when obvious.

## Anti-pattern — what NOT to do (real case)

The seven legacy lessons below were generated with **every** `correct` set to `0` (all answer `A`) and did not vary the slot — a CodeRabbit reviewer flagged it on the upstream PR. Treat these as the canonical "what not to do":

```text
07/15-attention-variants        10/15-speculative-decoding-eagle3
07/16-speculative-decoding      10/16-differential-attention-v2
08/19-visual-autoregressive-var 10/17-native-sparse-attention
                                10/34-gradient-checkpointing
```

Their positions were mechanically fixed, but their **content** is unverified — they are queued as `redo_quiz` in `manifest.json`. When you process a `redo_quiz` row, rebuild the questions from `docs/en.md` + `code/` and apply the variance rule; do not trust the existing text.
