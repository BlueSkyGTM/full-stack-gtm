# Helix Ramp Schedule
<!-- Stage 00-d output | 2026-06-12 -->

## What This Is

Helix's capabilities are introduced gradually across the 20-phase curriculum. A student in Phase 01 (Python environment setup) does not need full FSRS-driven recall scheduling on Day 1. A student in Phase 20 (capstone) should have the full system active.

This schedule defines exactly what Helix can do at each phase. Stage 06 uses this to implement the gradual introduction in the site UI — buttons appear, modalities unlock, and the FSRS review session becomes available at the right moment.

## Capability Levels

| Level | Name | What Helix can do |
|-------|------|------------------|
| L1 | Tutor only | ASSESS → EXPLAIN, HINT, CORRECT, ORIENT, REDIRECT. No quiz scoring, no FSRS. |
| L2 | Tutor + quiz scoring | L1 + QUIZ modality. Helix presents cards, accepts ratings, scores them. FSRS not yet active (no scheduling). |
| L3 | Tutor + FSRS scheduling | L2 + FSRS intervals computed and stored. Cards have `due` dates. Review sessions available. |
| L4 | Tutor + FSRS + copy-paste flag | L3 + FLAG CHECK modality. Helix parses copy-paste exercise output. |
| L5 | Full (L4 + Revelation 1) | L4 + Revelation 1 trigger available. Helix can surface the "you've seen this before" pattern across phases. Requires cross-phase student state read. |

## Per-Phase Schedule

| Phase | Level | Notes |
|-------|-------|-------|
| 01 | L1 | Environment setup — tutor only. FSRS would feel premature when the student is installing Python. |
| 02 | L1 | Data structures — tutor only. Building the foundation before recall layer activates. |
| 03 | L2 | Web scraping — quiz scoring activates. Student has enough vocabulary to be tested. |
| 04 | L2 | Data pipelines — quiz scoring continues. FSRS not yet active. |
| 05 | L3 | LLM prompting — **FSRS activates.** This is the first GTM-primary phase. Recall matters here: prompt templates and testing protocols need to stick. |
| 06 | L3 | Embeddings — FSRS continues. Copy-paste exercises exist but flag parser not yet active. |
| 07 | L4 | Fine-tuning → ABM — **copy-paste flag activates.** Phase 07 exercises require students to run signal detection scripts. The flag confirms they ran the code. |
| 08 | L4 | Vector databases — copy-paste continues. |
| 09 | L4 | Agents — copy-paste continues. |
| 10 | L4 | Multi-agent — copy-paste continues. |
| 11 | L4 | Evaluations → Revenue intelligence — full L4. |
| 12 | L4 | Observability — full L4. |
| 13 | L4 | Deployment — full L4. |
| 14 | L4 | Cost optimization — full L4. |
| 15 | L4 | Security — full L4. |
| 16 | L4 | Distributed systems — full L4. |
| 17 | L4 | MLOps — full L4. |
| 18 | L4 | Advanced prompting — full L4. |
| 19 | L5 | RAG → **Revelation 1 trigger available.** Student can now surface connections across all 18 completed phases. Helix can say "this pattern first appeared in Phase 03." |
| 20 | L5 | Capstone — full system. FSRS, flag parsing, and Revelation 1 all active. |

## Revelation 1 Trigger (L5)

Revelation 1 is Helix's cross-phase bridge capability. When active, Helix can:
- Reference a concept the student learned in an earlier phase: "You saw this stability/decay model first in Phase 05 — this is the same pattern applied to GTM signal decay."
- Surface unexpected connections between AI concepts and GTM playbooks
- Flag when a student is re-encountering a concept they already have FSRS cards for ("You have 3 cards on this already — want to review them now, or continue?")

Revelation 1 requires access to the student's full FSRS state (not just the current card). This is why it activates at Phase 19 — by then the student has enough review history for the connections to be meaningful.

## UI Implementation Notes (Stage 06)

- L1 → L2 transition: "Quiz" button appears in the Helix UI
- L2 → L3 transition: "Review session" option appears (shows due card count)
- L3 → L4 transition: "Paste output" box appears in exercise UI
- L4 → L5 transition: "Connections" link appears in Helix header (Revelation 1 entry point)

Each transition is gated by `currentPhase >= threshold` in the client state. No server-side gating required for L1–L4. L5 requires cross-phase FSRS state read — see `student-state-options.md` for the mechanism decision.

## What Helix Does Before Its Level Activates

If a student in Phase 01 (L1) asks Helix to quiz them, Helix responds:

> "Quiz scheduling activates in Phase 03. For now, I can explain concepts, give hints, and help you work through exercises. What are you stuck on?"

No error. No broken state. Clean redirect to an available modality.
