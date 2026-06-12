# Quality rubric — PM spot-check

Use when reviewing one quiz per phase after factory runs. Score **Pass / Revise** per row.

## A. Alignment (source of truth)

| # | Criterion | Pass if |
|---|-----------|---------|
| A1 | Every question maps to a Learning Objective or named concept section | You can point to a paragraph in `docs/en.md` |
| A2 | No facts that appear only in model prior or other lessons | Cross-lesson refs only where prerequisites list them |
| A3 | Build lessons: ≥1 check or post references code/demo | Names behavior from `code/` or Build It |
| A4 | Title matches lesson H1 / README | Consistent naming |

## B. Difficulty (not too easy)

| # | Criterion | Pass if |
|---|-----------|---------|
| B1 | pre is recall/understand, not “which letter is A” | Requires reading Problem/Concept |
| B2 | Three checks test **three different** sub-ideas | Not three rephrasings of one fact |
| B3 | At least one check requires mechanism or math from doc | Not all definition-level |
| B4 | post requires integration (two ideas or doc+code) | Single-sentence hook insufficient |
| B5 | Wrong options are plausible | A strong student could pick them before studying |

## C. Technical (site + CI)

| # | Criterion | Pass if |
|---|-----------|---------|
| C1 | 6 questions: pre, check×3, post×2 | `audit_lessons.py` clean |
| C2 | 4 options each, valid `correct` index | L008/L009 pass |
| C3 | All explanations non-empty (Tier B) | `--strict-quiz` pass for this file |
| C4 | No `placeholder` in option strings | L011 pass |
| C5 | No legacy keys | L007 pass |
| C6 | Answer key not constant; correct slot varies | not all six `correct` equal — L013 pass (`--strict-quiz`) |

## D. Tone

| # | Criterion | Pass if |
|---|-----------|---------|
| D1 | Matches course voice (technical, direct) | Not bloggy or exam-boilerplate |
| D2 | Explanations teach | Not “Correct!” only |

**Revise:** factory operator fixes quiz.json only; do not edit `docs/en.md` unless content is wrong independent of quiz.

Compare bar: [/lesson-planning](../.cursor/skills/lesson-planning/SKILL.md) (seven insights).
