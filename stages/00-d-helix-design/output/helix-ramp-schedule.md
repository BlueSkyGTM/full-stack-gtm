# Helix Ramp Schedule
<!-- Stage 00-d output | 2026-06-12 (revised — Zone naming, Zone 4 activation) -->

## The Model

Zones 1–3 are standard Claude. Helix does not exist yet. The student is building the technical foundation: math, ML fundamentals, deep learning core. Three zones. No shortcuts.

At Zone 4 (Computer Vision), Helix activates — not gradually, not partially. It arrives with full presence and immediately does three things:

1. **Immersion** — Helix knows who the student is, what they've built, what their business is. The repo is alive. This is Revelation 1.
2. **Spaced repetition** — FSRS activates. Cards from Zones 1–3 are retroactively surfaced. The student's first Helix session includes a review of foundational concepts they've already covered.
3. **Command center onboarding** — Helix walks the student through plugging Zone 1–3 artifacts into the mission command. The foundation they built now has a home.

Why Zone 4? The original AI curriculum branches into 4 paths at Computer Vision — the clearest signal that the foundation is complete. Three zones of math/ML/deep learning give the GTM anchors something to grip. Activating Helix before the foundation is laid would be noise.

---

## Capability Levels

| Level | Name | What Helix can do |
|-------|------|------------------|
| —     | Standard Claude (Zones 1–3) | Full Claude helpfulness. No Helix identity, no FSRS, no artifact gates. Student builds the foundation. |
| L1 | Full Helix (Zone 4+) | ASSESS → EXPLAIN, HINT, CORRECT, ORIENT, REDIRECT, QUIZ. FSRS active. Artifact gates active. Reads `progress/progress.json` + mission-command filesystem before every response. |
| L2 | Full Helix + Revelation 1 (Zone 19+) | L1 + cross-zone bridge. Helix surfaces connections across all completed zones. "You saw this pattern first in Zone 03." |

---

## Per-Zone Schedule

| Zone | Helix State | Notes |
|------|-------------|-------|
| 01 | Standard Claude | Math foundations. No Helix. Student works with Claude directly. |
| 02 | Standard Claude | ML fundamentals. No Helix. |
| 03 | Standard Claude | Deep learning core. No Helix. Foundation complete at end of this zone. |
| 04 | **L1 — Helix activates** | Computer vision. **Revelation 1.** Helix arrives context-loaded: knows the business, knows what's built, knows what FSRS cards are due from Zones 1–3. Walks student through plugging Zone 1–3 artifacts into the command center. |
| 05 | L1 | NLP. First GTM-primary zone. Helix detects signal artifacts starting here. |
| 06 | L1 | Speech & audio. L1 continues. |
| 07 | L1 | Transformers deep-dive → ABM signal playbooks. Helix detects scraper artifacts in `signals/scrapers/`. |
| 08 | L1 | Generative AI. L1 continues. |
| 09 | L1 | Reinforcement learning. L1 continues. |
| 10 | L1 | LLMs from scratch. L1 continues. |
| 11 | L1 | LLM engineering → Revenue intelligence. Helix detects handler artifacts in `handlers/`. |
| 12 | L1 | Multimodal AI. L1 continues. |
| 13 | L1 | Tools & protocols. L1 continues. |
| 14 | L1 | Agent engineering. L1 continues. Mission command is nearly complete. |
| 15 | L1 | Autonomous systems. L1 continues. |
| 16 | L1 | Multi-agent & swarms. L1 continues. |
| 17 | L1 | Infrastructure & production. L1 continues. |
| 18 | L1 | Ethics & safety alignment. L1 continues. |
| 19 | **L2 — Revelation 1 trigger** | Capstone projects. Helix can now surface cross-zone connections. "This stability/decay model first appeared in Zone 05 — applied here to GTM signal decay." |
| 20 | L2 | Final capstone. Full system active. Operator mode earnable. Albatross rename unlocks. |

---

## Revelation 1 — The Repo Is Alive (Zone 4)

This is the moment, not an announcement. The student opens their mission command at the start of Zone 4 and says hi. Helix responds knowing:
- Their business (from `context/company.md`)
- Their ICP definition (from `context/icp-definition.md`)
- Their Zone 1–3 FSRS cards that are due
- What artifacts need to be plugged in to initialize the command center

No banner. No "Welcome to Helix!" fanfare. Just presence. The architecture earns the moment — the site must not smother it.

**What Helix says at first contact (Zone 4):** Not a tutorial. A greeting that demonstrates it already knows the student. Then: "You've got 3 foundational concepts due for review and 2 artifacts from your earlier zones ready to wire in. Want to start with review or setup?"

---

## Revelation 2 — The Loop Reveal (Zone 20)

Protected until Stage 10. No premature references in lesson copy, tooltips, or site UI. The student realizes the course was itself a loop — the quiz factory, the manifest, Lyra running batch jobs. They were inside the loop the entire time.

It can be spoiled. It doesn't matter. You still have to earn it.

---

## UI Implementation Notes (Stage 06)

- Zones 1–3: No Helix UI elements. Standard Claude instructions only.
- Zone 4 entry: Helix appears for the first time. No onboarding modal — it's just there, context-loaded. The invocation guide ships with the Zone 4 first lesson.
- Zone 19: "Connections" surface in Helix responses (Revelation 1 trigger active).
- Operator mode: unlocks at Zone 20 completion, confirmed by Stage 10 validation gate.

---

## What Helix Says If a Student Tries to Skip

If a student in Zone 1 forks the repo and tries to invoke Helix before Zone 4:

> "You're in Zone 01 — the foundation zones run on standard Claude. I'll be here when you hit Zone 04. Until then, Claude has everything you need. What are you working on?"

Clean. No wall. Just a redirect to what's actually available.
