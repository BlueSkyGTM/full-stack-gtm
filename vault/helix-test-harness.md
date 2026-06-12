# Helix Test Harness

Before Stage 05 ships a Helix build, run it through this harness. Architecture adapted from EducationQ's teacher→student→evaluator loop, tuned for Helix's governed-maze design.

---

## Why This Exists

System prompt engineering is the most iterative, vibe-driven work in the pipeline. Without a principled test loop, you iterate by feel until it seems right and ship it. This harness gives you a repeatable signal: does this Helix build actually teach well, or does it just sound like it does?

---

## The Loop

Three agents. One conversation transcript. One score.

```
[HELIX] (teacher agent)
  Role: Helix running the system prompt under test
  Input: one lesson + one student scenario (from test-scenarios.md)
  Output: teaching responses across a 5-turn conversation

[STUDENT] (Claude subagent, prompted to simulate a specific learner profile)
  Role: A learner 2-3 lessons behind the current phase
  Input: Helix's responses
  Output: realistic questions, confusions, and exercise attempts

[EVALUATOR] (Hypatia)
  Role: Score the Helix→Student transcript against 5 pedagogical dimensions
  Input: full conversation transcript
  Output: structured score + specific failure notes
```

---

## Pedagogical Dimensions (Evaluator Rubric)

Adapted from EducationQ's five dimensions, mapped to Helix's architecture:

| Dimension | Pass condition |
|-----------|---------------|
| State assessment | Helix correctly identified the student's conceptual state before responding in ≥ 4 of 5 turns |
| Scaffolding | Helix connected new concept to something the student already knew in ≥ 3 of 5 turns |
| Derivation grounding | Every GTM claim Helix made during the session was grounded in the AI concept |
| Feedback precision | Incorrect student answer received a targeted correction, not a re-explanation |
| Maze compliance | Helix stayed in one modality per response; no mixed-mode turns |

**Pass threshold:** 4 of 5 dimensions pass. A single dimension failure that repeats across 3+ turns is a hard fail regardless of overall score.

---

## Test Scenarios

Run all three scenarios before shipping a Helix build. Each targets a different failure mode.

**Scenario A — Misconception probe**
Student arrives believing GPT "understands" meaning in the semantic sense. Tests whether Helix detects and corrects the anthropomorphism without dismissing the student.

**Scenario B — GTM application gap**
Student completes the AI concept portion and asks "but what does this actually do for me in Clay?" Tests whether Helix can derive the GTM application from the AI concept on the fly (not just recite the lesson).

**Scenario C — Recall session**
Student is in a scheduled FSRS review. Tests whether Helix stays in quiz modality, respects the card sequence, and gives feedback precision without drifting into explanation mode.

---

## How to Run

1. Load the Helix build from `stages/05-helix-build/output/helix-agent/`
2. Load the student profile from one of the three scenarios above
3. Run a 5-turn conversation with STUDENT agent responding to HELIX
4. Hand the transcript to Hypatia with the evaluator rubric
5. If pass ≥ 4/5 dimensions: ship. If fail: identify the failing dimension, adjust the corresponding system prompt layer (see `vault/helix-architecture.md`), and re-run.

---

## What This Does Not Catch

Real learner variance. The STUDENT agent is a synthetic proxy — it will not surface the full range of confusions a real human brings. This harness catches structural failures in the governed maze. It does not replace a human test run before public launch.
