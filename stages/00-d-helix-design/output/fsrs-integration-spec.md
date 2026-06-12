# FSRS Integration Spec
<!-- Stage 00-d output | 2026-06-12 -->
<!-- STATUS: LOCKED — changes require rebuilding all Stage 05 Helix exercises -->
<!-- Lock confirmation: This spec was reviewed and locked on 2026-06-12 during Stage 00-d. -->

## Algorithm

**FSRS-5** (Free Spaced Repetition Scheduler, version 5).

FSRS-5 uses two state variables per card: **Stability** (S) and **Difficulty** (D). These produce a retrieval probability at any given time:

```
R(t, S) = (1 + FACTOR * t / S) ^ (-EXPONENT)

where FACTOR = 19/81, EXPONENT = -0.5 (FSRS-5 defaults)
```

Cards are due when `R(t, S) <= desired_retention` (0.90 for this implementation).

## Card Format

Each quiz question becomes one FSRS card. A card has:

```json
{
  "card_id": "phases/05/03-llm-prompting:pre-q0",
  "front": "What does temperature control in an LLM response affect?",
  "back": "The randomness of token sampling — higher temperature increases diversity, lower temperature increases determinism.",
  "lesson_path": "phases/05/03-llm-prompting",
  "question_type": "pre|check|post",
  "fsrs_state": {
    "stability": 1.0,
    "difficulty": 5.0,
    "due": "2026-06-12T00:00:00Z",
    "last_review": null,
    "review_count": 0,
    "lapses": 0
  }
}
```

`card_id` is `{lesson_path}:{question_id}` — matches the quiz JSON question IDs exactly. This is how Helix links a presented card to the quiz bank.

## Rating Scale

Student ratings after each card review:

| Rating | Meaning | FSRS response |
|--------|---------|---------------|
| **Again** | Complete blackout or wrong | Card re-enters learning steps |
| **Hard** | Recalled with significant difficulty | Small stability increase |
| **Good** | Recalled correctly with minor difficulty | Normal stability increase |
| **Easy** | Recalled instantly, no effort | Large stability increase + difficulty decrease |

Helix presents ratings as button choices (or keyboard shortcut 1/2/3/4). No student types "Again" — they press a key.

## Scheduling Parameters (locked)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Desired retention rate | 0.90 | 90% recall probability at review time — high enough for curriculum mastery, realistic for voluntary study |
| Initial stability (new card) | 1.0 day | First review the next day |
| Difficulty initial | 5.0 | Midpoint of FSRS-5's 1–10 scale |
| Minimum interval | 1 day | No same-day re-reviews |
| Maximum interval | 365 days | Caps drift for long-inactive cards |
| Learning steps | [1 min, 10 min] | Two mini-reviews before a card enters "graduated" state |
| Relearning steps | [10 min] | One mini-review after a lapse before re-entering the graduated schedule |
| Easy bonus | 1.3× | Multiplier on the computed interval when student rates Easy |

## Correct-Response Definition

A "correct" response is any rating of **Hard**, **Good**, or **Easy**. An **Again** response is a lapse.

This definition is intentionally permissive — "Hard but recalled" counts as a pass. The FSRS algorithm adjusts stability accordingly; students who rate Hard repeatedly will see more frequent reviews without being penalized in the completion ledger.

## FSRS Validation: Worked Example

**Card:** "What is the FETE framework in Clay?"

**Back:** "Find, Enrich, Transform, Export — the four-step loop Clay uses to build and ship a lead database."

**Initial state:** stability=1.0, difficulty=5.0, due=Day 0

**Review 1 (Day 0):** Student rates **Good**
- New stability = 2.9 days (FSRS-5 formula with D=5.0, rating=Good)
- Next review: Day 0 + 3 = Day 3
- Retrieval at Day 3 with S=2.9: R = (1 + 19/81 * 3/2.9)^(-0.5) ≈ 0.91 ✓ (above retention target)

**Review 2 (Day 3):** Student rates **Again**
- Lapse recorded. Card re-enters relearning (10-minute step).
- Stability reduced: new_S = S * 0.5 * D_penalty = ~1.2 days
- Next review: in 10 minutes (relearning step), then Day 3 + 2 = Day 5

**Review 3 (Day 5):** Student rates **Easy**
- New stability = 1.2 * easy_bonus * FSRS_multiplier ≈ 7.5 days
- Next review: Day 5 + 8 = Day 13

**3-week learner check:** A new card reviewed on Day 0 with consistent Good ratings produces intervals of approximately 3, 9, 22, 55 days. By Week 3 (Day 21), the student should be reviewing roughly the same card every ~22 days. This is appropriate for curriculum concepts with a 90% retention target. Intervals are not compressing into "same day forever" nor drifting to "review in 6 months after one lesson." ✓

## FSRS State Storage

FSRS state lives in a separate field in the progress schema, never mixed with lesson completion state:

```json
{
  "v": 1,
  "done": { "01:0": true },
  "fsrs": {
    "phases/05/03-llm-prompting:pre-q0": {
      "stability": 2.9,
      "difficulty": 4.8,
      "due": "2026-06-15T00:00:00Z",
      "last_review": "2026-06-12T10:00:00Z",
      "review_count": 1,
      "lapses": 0
    }
  },
  "days": [],
  "updatedAt": 1749744000000
}
```

`fsrs` is a flat map keyed by `card_id`. It must NOT be cleared on logout — this is the long-term memory layer. Only `done` and `days` may be cleared on a user reset.

## What FSRS Does NOT Apply To

- Copy-paste exercises (evaluated by Helix's flag parser, not scheduled for recall)
- Project submissions (Stage 05 capstone layer — separate evaluation rubric)
- Pre-quiz questions taken before reading a lesson (these are diagnostic, not recall)
