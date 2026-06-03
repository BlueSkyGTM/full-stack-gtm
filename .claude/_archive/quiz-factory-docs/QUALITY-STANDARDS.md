# Quiz Quality Standards

Formal, scored extension of `QUALITY-RUBRIC.md`. Use when grading a `quiz.json` during factory review, PM spot-check, or corpus audit. Every criterion in `QUALITY-RUBRIC.md` is preserved and extended here with numeric scores, severity tags, and pass thresholds.

---

## Scoring system

| Score | Meaning |
|-------|---------|
| **2** | Pass — fully meets the criterion |
| **1** | Warn — partially meets; fix before next phase ships |
| **0** | Fail — does not meet; severity determines urgency |

**Maximum score:** 18 criteria × 2 = **36 points**

Breakdown: A (4 criteria, 8 pts) + B (7 criteria, 14 pts) + C (5 criteria, 10 pts) + D (2 criteria, 4 pts) = 36 pts.

**Grade thresholds per quiz:**

| Grade | Score | % | Meaning |
|-------|-------|---|---------|
| A | ≥ 33 / 36 | ≥ 92 % | Ready to ship; only minor warn items allowed |
| B | ≥ 28 / 36 | ≥ 78 % | Acceptable; warn items queued for the next pass |
| C | ≥ 23 / 36 | ≥ 64 % | Needs revision before the lesson goes live |
| D | < 23 / 36 | < 64 % | Rebuild — queue as `redo_quiz` in manifest |

**Phase coverage target:** ≥ 80 % of audited lessons grade B or higher before Claude Code advances to the next phase in the manifest.

---

## Severity classification

| Severity | Tag | Meaning | Required action |
|----------|-----|---------|-----------------|
| Ship-blocker | **P0** | Wrong answer, wrong doc-mapping, schema invalid | Fix before any learner sees the lesson |
| Fix-before-live | **P1** | Misleading distractor, empty explanation, one check tests same idea as another | Fix before phase ships |
| Improve-later | **P2** | Tone off, option text could be clearer, B6/B7 border cases | Queue for a polish pass |

A single P0 finding caps the quiz at grade C regardless of total score.

---

## A. Alignment (source of truth)

These criteria check that every question, correct answer, and explanation maps to the lesson's own document — not model prior, not other lessons.

| # | Criterion | Score 2 | Score 1 | Score 0 | Severity if 0 |
|---|-----------|---------|---------|---------|----------------|
| A1 | Every question maps to a Learning Objective, named concept section, or Build step in `docs/en.md` | Can point to exact paragraph for all 6 | 5/6 have a source | Any question has no doc mapping | P0 |
| A2 | No facts from model prior or unlisted external sources | All claims come from `docs/en.md` or `code/` | One minor fact is general knowledge but not wrong | Any fact contradicts or fabricates doc content | P0 |
| A3 | Cross-lesson refs only where `docs/en.md` prerequisites list them | Zero cross-phase imports; prereqs cited only when listed | One unlisted cross-phase ref that is harmless context | Question tests a concept from an unlisted prerequisite lesson | P1 |
| A4 | Build lessons: ≥ 1 check or post question references `code/` or the Build It demo | At least one question names a function, step, or observable output from `code/main.*` | Only a vague "as the code shows" reference, not specific | Build lesson with no code reference at all | P1 |

---

## B. Difficulty calibration

These criteria check that questions sit at the right cognitive level per stage, that three check questions are genuinely distinct, and that wrong options require real study to reject.

| # | Criterion | Score 2 | Score 1 | Score 0 | Severity if 0 |
|---|-----------|---------|---------|---------|----------------|
| B1 | `pre` is recall / understand (not trivial guessing) | Requires reading the Problem / hook; not answerable from the title alone | Answerable from common knowledge but not trivially | A naive student with no reading can guess correctly without thinking | P1 |
| B2 | Three `check` questions test **three different** sub-ideas | Each check targets a distinct mechanism, formula, or tradeoff from the doc | Two checks are closely related (different phrasing, same idea) | Two or more checks are rephrasing of the same single fact | P1 |
| B3 | At least one `check` requires mechanism or math from the doc | One check applies a formula, algorithm step, or quantitative tradeoff | All checks are at definition level but well-written | All checks are pure vocabulary recall with no application | P1 |
| B4 | Both `post` questions require integration | Each post integrates two or more doc ideas, or connects doc + code | One post integrates; one is recall-level | Both posts are single-fact recall | P1 |
| B5 | Wrong options are plausible before studying | A prepared student who skimmed the topic would find at least one distractor tempting | One distractor per question is clearly wrong (too easy) | Any option is trivially absurd or a joke answer | P2 |
| B6 | **Vocabulary precision:** technical terms match the lesson's Key Terms table | All technical terms used in questions match the exact wording in the Key Terms table or the doc's section headers | Minor paraphrase of a term (not misleading) | A technical term is used in a different sense than the doc defines | P1 |
| B7 | **Distractor authenticity:** wrong options are mistakes the doc explicitly refutes | For each question, the wrong options correspond to misconceptions the doc's body directly addresses | Some wrong options are plausible but not explicitly refuted by the doc | A wrong option is a random wrong fact never discussed in the lesson | P2 |

---

## C. Technical / schema

Automated by `scripts/audit_lessons.py` (Tier A) and `--strict-quiz` (Tier B). Included here for completeness.

| # | Criterion | Score 2 | Score 1 | Score 0 | Severity if 0 | Auto-checked |
|---|-----------|---------|---------|---------|----------------|--------------|
| C1 | 6 questions: exactly 1 pre, 3 check, 2 post in that order | Passes L001–L006 | — | Wrong count or order | P0 | Yes (Tier A) |
| C2 | 4 options each; `correct` is valid zero-based index | Passes L008/L009 | — | Invalid index or wrong option count | P0 | Yes (Tier A) |
| C3 | All 6 explanations non-empty and substantive | All ≥ 20 chars, no "Correct!" only | One explanation is thin but present | Any explanation empty or whitespace-only | P1 | Yes (Tier B `--strict-quiz`) |
| C4 | No `placeholder` in option strings | Passes L011 | — | Any option contains the word `placeholder` | P1 | Yes (Tier B) |
| C5 | Answer key is not constant (variance rule) | At least 3 distinct correct indices across 6 questions | Only 2 distinct indices (warn) | All 6 correct indices identical | P1 | Yes (Tier B L013) |

---

## D. Voice and tone

| # | Criterion | Score 2 | Score 1 | Score 0 | Severity if 0 |
|---|-----------|---------|---------|---------|----------------|
| D1 | Matches course voice: technical, direct, no exam boilerplate | No "Which of the following…", no "All of the above" (unless positionally required), no filler intros | One filler phrase | Multiple boilerplate openings or bloggy hedging throughout | P2 |
| D2 | Explanations teach, not just confirm | Each explanation adds a sentence the question did not already contain; cites the doc mechanism | Explanation mostly restates the question stem | Any explanation is "Correct!" or one word only | P1 |

---

## Scoring worksheet (per quiz)

```
Section A:  A1__ + A2__ + A3__ + A4__                        = __ / 8
Section B:  B1__ + B2__ + B3__ + B4__ + B5__ + B6__ + B7__  = __ / 14
Section C:  C1__ + C2__ + C3__ + C4__ + C5__                 = __ / 10
Section D:  D1__ + D2__                                      = __ / 4
                                                     TOTAL   = __ / 36

Grade:  A (≥33/36)  B (≥28/36)  C (≥23/36)  D (<23/36)
P0 found? → cap at C regardless of score
```

---

## Quick reference: P0 checklist

Before submitting any `quiz.json`, verify none of these apply:

- [ ] Correct answer is wrong per `docs/en.md`
- [ ] A question has no identifiable source paragraph in `docs/en.md`
- [ ] Stage sequence is not exactly `pre, check, check, check, post, post`
- [ ] `correct` index is out of range (≥ number of options)
- [ ] Question or option text contradicts a claim in `docs/en.md`

Any single check = P0 = grade cap C = must fix before lesson ships.

---

## Relationship to other files

| File | Role |
|------|------|
| `QUALITY-RUBRIC.md` | Human spot-check shorthand (subset of these criteria) |
| `REFERENCES.md` | Schema, variance rule, difficulty bar, anti-patterns |
| `ARCHITECTURE.md` | How to generate a quiz step by step |
| `CLAUDE.md` | Operator rules for Claude Code batch runs |
| `scripts/audit_lessons.py` | Automates all C-section criteria |
| `.cursor/skills/lesson-planning/SKILL.md` | Seven insights, flagship bar, triage guide |
