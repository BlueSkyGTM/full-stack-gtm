# BlueSkyGTM Engineering

GTM engineering with a technical foundation. Free and open source.

**The curriculum teaches you to build a live GTM system — signal detection, AI infrastructure, outbound execution — using the same agentic tooling you learn about along the way.**

---

## What This Is

BlueSkyGTM Engineering is a 20-phase curriculum for people who want to be GTM engineers, not just GTM tool users. The technical depth is intentional: when the tool stack turns over (it will), you need to understand what the tools are actually doing.

The course covers:
- AI engineering fundamentals (transformers, embeddings, FSRS, attention)
- GTM application (signal scoring, ICP modeling, outreach infrastructure)
- A full GTM Starter Kit — your own signal engine, scraper handlers, and mission command repo

By the end, you have a working GTM system configured for your real business, not a certificate.

**Platform:** Claude Code Desktop. One environment — the terminal. No IDE switching, no browser-dependent exercises. This is how modern AI engineers actually work.

---

## The Double Helix

The curriculum is structured around two interlocking tracks:

- **AI Engineering** — the reasoning chain. Transformers, attention, embeddings, FSRS, fine-tuning, agent architecture.
- **GTM Engineering** — the application. Every AI concept maps to a concrete GTM mechanism: signal scoring, enrichment pipelines, ICP clustering, outreach drafting.

Neither track is filler for the other. The AI engineering explains *why* the GTM tools work. The GTM application gives the AI engineering a reason to exist.

---

## Helix

Helix is the AI tutor built into the course. It runs inside Claude Code Desktop, reads your actual progress and business context, and adapts as you build.

**Early phases:** Helix is a quiz and recall tool.

**After Phase 04–05:** Helix reads your mission command repo — your ICP, your signal config, your company context — and starts answering operational questions, not just curriculum ones. No announcement. It is just there, ready, knowing your business. You will notice it.

**After graduation:** Helix switches to operations mode (RESEARCH / SCORE / BRIEF / DRAFT / STATUS / ESCALATE). Same governed system, different modality.

---

## The Mission Command

The GTM Starter Kit (`gtm-mission-command`) is a separate repo you fork at the start. It contains:

```
context/          company.md, ICP.md, signals.md, playbooks/
signals/          scrapers/, processors/, examples/
handlers/         research-handler, score-handler, outreach-handler
state/            STATE.md
CLAUDE.md
```

Your fork is the save file. Progress is stored in `progress/progress.json`, written by Helix, committed by you. No cloud sync. Your fork travels with you.

**The business registration is the new game.** Before Helix routes operational requests, `context/company.md` must have no placeholder values. Configure a real business — your own, or one you are building — to unlock the course.

This also solves the anti-cheat problem structurally: cloning someone else's completed repo gives you their business, their ICP, their scrapers configured for their signals. There is nothing to gain. Progress is only meaningful in the context of the business you configured.

---

## The Albatross

Your mission command is called the Albatross. When you complete the course and earn operator mode, two things unlock:

1. You can rename your mission command. Whatever you call it, that name propagates through your `CLAUDE.md`, your `STATE.md`, and Helix's self-references.
2. You can rename Helix.

The gate is hard. You earn it by running the system against a real business, not by finishing the lessons.

---

## Free. Actually Free.

The curriculum is free. Not "free tier with a paywall at lesson 12" free. Free.

The going rate for courses like this is $3,000 minimum. That is not this. If you are broke and deciding between boba and a course, have the boba. Come back when you want to. The course will be here.

The model: free curriculum, open source, Skool community for the people who want to build alongside others. If you want to support the work — Patreon, sponsorships, or hire BlueSkyGTM for private sector consulting. That is where the money comes from. Not from the people trying to break in.

Inspired by Second Wind. Not a coincidence they use a bird too.

---

## The Name

"Blue sky" research is theoretical work done without a specific application — the kind of thinking that happens before anyone knows it matters. The name comes from a period of unemployment spent working out theories about AI context management, with no job, no credentials, and no idea the field was about to become what it became. It turned out to matter.

The domain is `blueskygtm.engineer`. That is not aspirational. It is the job title the curriculum produces.

This course exists for people in that same position: technically capable, self-taught, not yet credentialed, and more ready than the market currently recognizes.

---

## Build Pipeline

This repo is both the course and the coursemaker. The build pipeline (Stages 01–10) generates the 498 lessons, exercises, and quizzes using Claude Code and a set of specialized agents (Lyra, Newton, Echo, Hypatia).

| Stage | What It Does |
|-------|--------------|
| 00-a–f | Archaeology, topic mapping, agent briefing, Helix design, vault bootstrap, tooling |
| 01 | Generate 498 lesson outlines + manifest (resumable queue) |
| 02 | Inject full hybrid lessons — chain-of-reason derivation, GTM weave, inline citations |
| 03 | Design CLI exercises that run entirely in Claude Code Desktop |
| 04 | Generate FSRS-formatted recall cards |
| 05 | Build Helix from the governed-maze spec |
| 06 | Site readability overhaul + pacing map + illustration generation |
| 07 | Student state persistence — repo-as-save-file, gate-check-spec |
| 08 | Wire handlers into the signal engine |
| 09 | Quality pass: CLARITY, WEAVE, ACCURACY judges on all 498 lessons |
| 10 | End-to-end validation — cold student through full path |

```
vault/                  course identity, voice, variables
stages/00-a → 00-f/     Phase 0
stages/01-10/           Build pipeline
references/             Loop engineering, signal engine architecture
shared/                 Quality standards, illustration pipeline
```

See `CLAUDE.md` for routing. See `vault/phase0-plan.md` to understand where the build currently stands.

---

## Status

Phase 0 in progress. Build pipeline fully designed and staged. 128 lessons built. 498 target.

The curriculum is being written by building it. The course author is the first student.

---

## License

MIT.
