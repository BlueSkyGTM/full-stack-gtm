# GTM Starter Kit — Course Integration Guide

Source: github.com/KarlRaf/gtm-starter-kit (The Revenue Architects / Synapse Academy)

---

## What It Is

The GTM Starter Kit is a Claude Code operating environment for GTM engineers. It is the tool students use to run their GTM practice — not a lesson, not a tutorial. Fork it once, fill in the context files, and every AI-powered GTM task runs from a one-line prompt.

This is not built into the course. It runs alongside it.

---

## How It Enters the Course

**Fork at course start.** Students fork the repo before Phase 01. Every phase's exercise updates one context file or extends one skill file. By Phase 20 the kit reflects 20 phases of real GTM practice — a live operating environment, not a toy project.

The kit is the **student's own context layer**. The curriculum teaches the AI engineering that makes it work. These are two separate repos serving different purposes.

---

## The Five Skills

| Skill | File | What it does | Curriculum connection |
|-------|------|-------------|----------------------|
| Setup | `skills/setup/SKILL.md` | Provisions all context files from a domain | Phase 01 — ICP & TAM |
| Account Research | `skills/account-research/SKILL.md` | Deep research brief before outreach | Phase 03 — Signal Detection |
| Signal to Sequence | `skills/signal-to-sequence/SKILL.md` | Turns a signal into a live outreach campaign | Phase 05 — Outbound Systems |
| ICP Scoring | `skills/icp-scoring/SKILL.md` | Scores account lists against ICP definition | Phase 02 — Lead Scoring |
| Weekly Update | `skills/weekly-update/SKILL.md` | Keeps context files current (the MLOps retraining loop for GTM) | Phase 17 + MLOps appendage |

---

## The Context Files

Six files that represent everything Claude needs to know about a student's GTM motion. Filled in once, kept current via weekly-update.

| File | What it holds | Updated by |
|------|--------------|-----------|
| `context/profile.md` | Company overview, product, team, customers | setup + weekly-update |
| `context/icp-definition.md` | ICP tiers, qualification criteria, anti-ICP filters | setup + weekly-update |
| `context/signal-library.md` | Every signal with detection method, scoring weight, decay logic, performance history | weekly-update + sync scripts |
| `context/positioning.md` | Value pillars, messaging matrix, what not to say | setup + weekly-update |
| `context/competitor-radar.md` | Battlecards, win/loss patterns, competitor movement | weekly-update |
| `context/personas/` | One file per buyer role — priorities, objections, how they buy | setup + weekly-update |

---

## The Signal Library as Phase 19/20 Capstone Target

The `signal-library.md` format includes decay logic for each signal — detection method, scoring weight, and performance history. This is the same stability/decay math as FSRS.

The Phase 19/20 capstone can port the FSRS algorithm to the signal library: instead of scheduling card reviews based on memory decay, it schedules account resurfacing based on signal decay. A Series B announcement has a half-life. A new RevOps hire goes cold after 14 days. FSRS can model when to re-engage.

This is the "weaponize Helix" direction — tuning an education algorithm into a GTM tool.

---

## Sync Scripts

Two Python scripts pull live outbound performance data back into the context files:

- `sync/sync-campaign-results.py` — pulls reply rates and meeting rates from Smartlead/Apollo, updates signal-library.md
- `sync/sync-signal-performance.py` — aggregates signal performance across campaigns

These require API keys in `.env`. Run them before weekly-update to ensure current performance data.

---

## Reference Implementation

`examples/sample-company/` — a fully built-out version for Relay (fictional workflow automation platform). The `signal-library.md` in the sample company is the reference for what a mature signal library looks like after six months of weekly updates: detection methods, decay logic, performance benchmarks, signal combination rules.

---

## Quick Start for Students

```bash
git clone https://github.com/KarlRaf/gtm-starter-kit.git && cd gtm-starter-kit && claude .
```

Then: `Read skills/setup/SKILL.md and set up this repo for [your-domain.com]`

That's the first exercise for Phase 01.
