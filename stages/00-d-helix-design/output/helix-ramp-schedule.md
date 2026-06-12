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
| L3 | Tutor + FSRS + artifact awareness | L2 + FSRS intervals computed and stored. Cards have `due` dates. Helix reads the mission-command filesystem to detect artifact gates. A student who says "hi" gets a response grounded in what's built and what's due. |
| L4 | Full (L3 + Revelation 1) | L3 + Revelation 1 trigger available. Helix can surface "you've seen this before" connections across phases. Requires cross-phase student state read. |

## Per-Phase Schedule

| Phase | Level | Notes |
|-------|-------|-------|
| 01 | L1 | Environment setup — tutor only. FSRS would feel premature when the student is installing Python. |
| 02 | L1 | Data structures — tutor only. Building the foundation before recall layer activates. |
| 03 | L2 | Web scraping — quiz scoring activates. Student has enough vocabulary to be tested. |
| 04 | L2 | Data pipelines — quiz scoring continues. FSRS not yet active. |
| 05 | L3 | LLM prompting — **FSRS + artifact awareness activates.** First GTM-primary phase. Helix now reads the mission-command filesystem; student's first real GTM artifacts start appearing here. |
| 06 | L3 | Embeddings — L3 continues. |
| 07 | L3 | Fine-tuning → ABM — Helix detects signal scraper artifacts in `signals/scrapers/`. |
| 08 | L3 | Vector databases — Helix detects handler artifacts in `handlers/`. |
| 09 | L3 | Agents — L3 continues. |
| 10 | L3 | Multi-agent — L3 continues. |
| 11 | L3 | Evaluations → Revenue intelligence — L3 continues. |
| 12 | L3 | Observability — L3 continues. |
| 13 | L3 | Deployment — L3 continues. |
| 14 | L3 | Cost optimization — L3 continues. |
| 15 | L3 | Security — L3 continues. |
| 16 | L3 | Distributed systems — L3 continues. |
| 17 | L3 | MLOps — L3 continues. |
| 18 | L3 | Advanced prompting — L3 continues. |
| 19 | L4 | RAG → **Revelation 1 trigger available.** Student can now surface connections across all 18 completed phases. Helix can say "this pattern first appeared in Phase 03." |
| 20 | L4 | Capstone — full system. FSRS, artifact gates, and Revelation 1 all active. Operator mode earnable. |

## Revelation 1 Trigger (L4)

Revelation 1 is Helix's cross-phase bridge capability. When active, Helix can:
- Reference a concept the student learned in an earlier phase: "You saw this stability/decay model first in Phase 05 — this is the same pattern applied to GTM signal decay."
- Surface unexpected connections between AI concepts and GTM playbooks
- Flag when a student is re-encountering a concept they already have FSRS cards for ("You have 3 cards on this already — want to review them now, or continue?")

Revelation 1 requires access to the student's full FSRS state (not just the current card). This is why it activates at Phase 19 — by then the student has enough review history for the connections to be meaningful.

## UI Implementation Notes (Stage 06)

- L1 → L2 transition: "Quiz" button appears in the Helix UI
- L2 → L3 transition: "Review session" option appears (shows due card count)
- L3 → L4 transition: "Connections" link appears in Helix header (Revelation 1 entry point)

Each transition is gated by `currentPhase >= threshold` in the client state. No server-side gating required. L4 requires cross-phase FSRS state read — state lives in `progress/progress.json` in the student's mission command fork.

## What Helix Does Before Its Level Activates

If a student in Phase 01 (L1) asks Helix to quiz them:

> "Quiz scheduling activates in Phase 03. For now, I can explain concepts, give hints, and help you work through exercises. What are you stuck on?"

No error. No broken state. Clean redirect to an available modality.
