# BlueSkyGTM — Course Polish & Power Plan

**Author:** Conductor (Claude/Opus)
**Director:** Raymond
**Mandate:** Make the course more engaging, the site less crappy, the lessons more powerful. Conductor orchestrates GLM fleet + sub-agents; director does not touch terminals.

---

## Diagnosis (verified ground truth, not the manifest's lies)

The manifest says "510 done." Reality, verified by reading the files:

| Defect | Count | % | Severity |
|--------|-------|---|----------|
| Lessons truncated (missing `## Sources`) | 270/510 | 52% | **CRITICAL** |
| Lessons cut off mid-code-block (unbalanced ```) | 151/510 | 29% | **CRITICAL** |
| Lessons with unfilled `[CITATION NEEDED]` | 383/510 | 75% | HIGH |
| Site serves UPSTREAM content (data.js → rohitg00 repo) | all | 100% | **CRITICAL** |
| Quiz bank built (Stage 04) | ~30/510 | 6% | HIGH |
| Mermaid rendering wired | no | — | MED |
| Pacing map / illustrations | none | — | MED |
| Dead auth/WP/Redis/RPG code live | present | — | LOW |

**Root cause of truncation:** Stage 02 dispatcher `max_tokens=8000`. Complete lessons exceed that and get cut mid-output. Confirmed: 151 end inside an open code fence. **This is a config bug. The methodology is sound** — complete lessons (`14-agent-engineering/01-the-agent-loop`, `07-transformers-deep-dive/02-self-attention-from-scratch`, `03-deep-learning-core/03-backpropagation`) are flagship-grade.

**The reframe:** the differentiator (the GTM weave) lives in the late "Use It"/"Ship It" sections — exactly the sections truncation amputates. So the missing 52% is disproportionately the GTM half. Finishing generation IS the highest-leverage act available.

---

## Phase A — Foundation repair (nothing else matters until this is done)

**A1. Fix + finish lesson generation.**
- Raise Stage 02 `max_tokens` 8000 → 16000.
- Add a completeness gate to the dispatcher: a lesson is only `done` if it contains `## Sources` AND has balanced code fences AND ≥ 8 section headers. Otherwise `failed`. (The current "done = a file exists" is why we shipped 270 broken lessons.)
- Regenerate the 270 truncated lessons via GLM-5.1 governed dispatch. Workers ≤ 3 (Z.ai breaks at 8).
- *Owner: Conductor writes the dispatcher fix + validator. GLM fleet regenerates.*

**A2. Wire hybrid-lessons → the site (the missing last mile).**
- Build `scripts/build_data.py`: reads `stages/02/output/hybrid-lessons/`, emits `site-new/js/data.js` with 510 lessons whose `url`/`path` point at THIS course's content, not upstream.
- Decide + implement serving: lessons fetched from this repo's raw GitHub path (or bundled at build). The reader must load OUR lesson, with OUR GTM strand.
- Verify end-to-end: open the reader, confirm a hybrid lesson with GTM content renders.
- *Owner: Conductor (critical architecture — done personally).*

---

## Phase B — Content power

**B1. Citation research pass.** GLM fleet resolves the 1098 `[CITATION NEEDED]` markers — one lesson per task, governed dispatch, real sources for GTM claims (Clay, Apollo, etc.). *Owner: GLM fleet, Taskmaster-managed.*

**B2. Mermaid lint pass.** Script + GLM fixup: validate every ```mermaid block parses; fix invalid syntax (e.g. `EV -->{...}` missing node id). *Owner: script-first, GLM for fixes.*

**B3. Cold-open hooks.** Add a one-sentence hook above "The Problem" in each lesson — a real deal-loss / debugging-failure moment. Converts "excellent textbook" → "can't stop reading." *Owner: GLM fleet.*

**B4. Finish Stage 04 quiz bank.** Diagnose the 40% failure rate (currently stalled), fix, complete all 510 cards.json. *Owner: Conductor diagnoses, GLM completes.*

---

## Phase C — A site that's genuinely impressive

**C1. Wire Mermaid.js + syntax highlighting (Shiki/Prism) + copy button** in the lesson reader. Kill the dead "View on GitHub" stub. *Owner: sub-agent, Conductor reviews.*

**C2. Typography + readability fix.** VT323 for display/hero ONLY; all lesson headers + body in JetBrains Mono. Add on-page TOC + scroll-spy + reading time; widen column to ~70ch, line-height 1.75. *Owner: sub-agent.*

**C3. Make the reader the front-end of Open-Brain (biggest missed opportunity).** At the bottom of each lesson, surface the FSRS cards that lesson generates, show a real forgetting-curve preview ("resurfaces in 3 days"), turn "Mark complete" into "Add to my repo's review deck." Converts the reader from a re-skinned blog into a live demo of the product being sold. *Owner: Conductor architects, sub-agent builds.*

**C4. Homepage interactive Open-Brain demo.** Replace the three static `<code>` lines with an animated FSRS scheduling visualization or a terminal-cast of the `clone → tutor picks up` loop. *Owner: sub-agent.*

**C5. Strip dead code.** Remove auth.js, WP/Redis adapters in store.js, HUD/XP/badge/player-card CSS (~230 lines), api/ backend. *Owner: sub-agent.*

---

## Phase D — Pipeline completion (after A–C land)

- D1. Pacing map (`stages/06/output/pacing-map.md`) — zone-by-zone difficulty curve. Feeds Stage 09.
- D2. Illustrations — Tier 1 (GLM-image) for concept beats, Tier 2 (Excalidraw) for architecture. Heavy; may phase.
- D3. Stage 07 (student-state gate checks), 09 (quality pass), 10 (validation run).

---

## Orchestration model

- **Conductor (me):** architecture, dispatcher fixes, the data.js wiring, Open-Brain reader integration, root-cause diagnosis, sub-agent orders, commits/pushes. The high-leverage work only.
- **GLM fleet (API):** bulk content regen, citation research, hooks, quiz completion — governed dispatch, ≤3 workers, completeness-gated.
- **Sub-agents (Taskmasters):** site build tasks, dead-code removal, scoped feature work — each gets a narrow target + one job.
- **Director (you):** approval gates only. Never a terminal.

## Sequencing rule

A blocks everything. A1 (content) and A2 (wiring) run in parallel. B and C run in parallel once A lands. D is last. Quality gate (`scripts/audit_lessons.py` + the new completeness validator) must pass before any "done."

## First three moves on approval

1. Patch Stage 02 dispatcher (max_tokens=16000 + completeness validator), kick the 270-lesson regen.
2. Personally build `scripts/build_data.py` and wire the reader to OUR content.
3. Spawn the site sub-agent fleet for Phase C1/C2/C5 (parallel, non-blocking).

---

## /autoplan RESULTS (4 independent reviewers — CEO / Design / Eng / DX, Codex absent)

The reviewers converged, without seeing each other, on the same critical faults. Cross-model consensus is the high-confidence signal.

### Auto-decided (mechanical — clearly correct, folded into plan v2)

| # | Fix | Why | Principle |
|---|-----|-----|-----------|
| 1 | **Reconciliation script before any regen.** Walk all 510 `en.md`, rewrite manifest `done`/`failed` by content (Sources + balanced fences + headers), NOT existence. | Manifest says all 510 "done" → `--retry-failed` finds 0; a blind re-run clobbers the 240 flagship lessons. This is the #1 data-loss risk. | Explicit |
| 2 | **Probe before batch.** Regen 1 lesson at max_tokens=16000, assert API `finish_reason=="stop"` (not `"length"`). Log finish_reason per job henceforth. | 16000 may exceed the provider's real output cap → re-truncate at a new ceiling, burn the whole budget. Dispatchers currently discard finish_reason. | Pragmatic |
| 3 | **Two-tier gate.** `structure-complete` (blocks write: balanced fences, 8 `^## ` headers, per-section min-chars) vs `ship-ready` (blocks publish: zero `[CITATION NEEDED]`, GTM-weave check, judge pass). | A `## Sources` header ≠ real sources; 75% still carry CITATION NEEDED and would pass a structural gate. | Completeness |
| 4 | **C3 reframe (the original is impossible).** A static no-auth site CANNOT read the student's local `progress/progress.json`. Reader shows the lesson's *static* `cards.json` ("this lesson seeds N cards") + "Add to deck" = a copyable terminal command. Forgetting-curve animation moves to the **homepage** (honest marketing, not faked student state). | C3 as written contradicts the homepage's own "no app, no account, state lives in your repo" promise 4×. Faking due-dates on a paid course is a trust-killer. | Explicit |
| 5 | **A2 is downstream of A1, not parallel.** Don't repoint/bundle a lesson URL until that lesson is regenerated + validated. | A2 consumes A1's output; parallel = bundling truncated lessons / 404s. | — |
| 6 | **B1 citations: not a bulk GLM dispatch.** GLM hallucinating a plausible Clay citation launders a fabrication into authority — worse than a visible marker on a PAID course. Resolve only lessons still containing the literal marker (idempotent); require verifiable sources; sample-audit. | Highest embarrassment-per-dollar line in the original plan. | — |
| 7 | **First-run seed (new, was missing entirely).** Ship a pre-seeded `progress/progress.json` + pre-`init`'d FSRS deck in the clone; rewrite the empty-clone greeting (hand over Zone-00-L01 URL, don't ask "what zone?"); add a prerequisites strip (Claude Code Desktop, git, Python). | Fresh clone today = dead-end greeting, Helix gated till Zone 04, no cards, no handoff. Cheapest high-leverage student fix; plan addressed zero student-runtime states. | Completeness |
| 8 | **Reading-mode reader surface.** Kill the background grid inside `.lz-article`, lift off pure-black, demote saturated red to one role (links), neutral inline-code. Keep 8-bit chrome outside the column. | Highest perceived-quality-per-hour move; the dark grid + all-red fights a 30-min technical read. | — |

### Surfaced to director (genuine decisions — see gate)

- **Scope strategy** (User Challenge — all 4 models): slice-first-and-validate vs full-foundation-repair-then-polish.
- **`phases/` cutover** (User Challenge — CEO): hybrid-lessons is canonical, `phases/` (+ its 351 quizzes, graph, quiz-factory, .claude rules) is dead. Confirm + audit blast radius.
- **Serving architecture** (Taste — Eng): bundle-at-build (recommended) vs runtime-raw-fetch. Current reader hardcodes `/main/` (repo is `master`) and a `phases/`-only path regex — both break against `stages/.../hybrid-lessons/` regardless.
