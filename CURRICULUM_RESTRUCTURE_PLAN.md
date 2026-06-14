# Curriculum Restructure — Two-Loop Architecture

**Director:** Raymond · **Conductor:** Claude (Opus) · **Date:** 2026-06-14
**Status:** Reviewed by /autoplan crew (CEO · Design · DX). Loop 1 ready to build; Loop 2 documented for later.

---

## 1. The Decision

One track, looped twice. Same 20 zones both times.

- **Loop 1 — Fundamentals (the 80/20 core).** Every concept across all 20 zones, lean and low-resistance. You understand the mechanism, build it from scratch once, and run a small real GTM slice with it. Fast to finish — the point is momentum.
- **Loop 2 — Intermediate (the 20/80 depth).** The same 20 zones turned up: the hard production applications, the full GTM pipelines, the scale-and-deploy work. For when you want the 20/80 regardless of effort.
- **Not isolated.** Loop 2 lives on the same site, expanded later. One curriculum that deepens, never a walled-off "advanced" section.
- **Artifact-meter between loops.** Building out your mission-command repo fills a visible "Loop 2 key" meter. Complete the fundamentals artifact and Loop 2 opens. The repo was always the save file; now it's also the meter. (Meter, not wall — see §4.5, this is the crew's main correction.)

The principle this corrects: most fundamentals courses are bloated sandboxes that make you grind excessive depth just to unlock the next sandbox. This one is breadth-complete but depth-deferred. Cover everything fast, then *choose* to go deep.

---

## 2. The Alignment Insight (why the timing is lucky)

The lessons that truncated got cut off in their **back half** — the "Ship It" production sections and heavy applications. That is *exactly* the half this restructure moves to Loop 2. The token-ceiling bug didn't randomly destroy content; it pre-sorted it along the cut line we'd have drawn anyway.

**Verified reconciliation (510 lessons, against the lean cut line):**

| Bucket | Count | % | Action |
|--------|-------|---|--------|
| **Lean-ready** (complete; just trim the heavy Ship It) | 238 | 46% | Trim → done |
| **Front-intact** (truncated in/after Use It; Concept+Build good) | 155 | 30% | Close + Sources (light) |
| **Mid-build** (died during Build It) | 78 | 15% | Medium completion |
| **Early-broken** (died before Build It) | 39 | 7% | Full regen |

**77% (393/510) need no regeneration** — they're lean-ready or a light touch. Only **117** need real generation work, **39** of them serious. "Regenerate 270 broken lessons" was the wrong job; the right job is ~4x smaller.

---

## 3. The Cut Line

| Lesson section | Loop 1 (Fundamentals) | Loop 2 (Intermediate) |
|----------------|:---:|:---:|
| Learning Objectives | ✅ (trimmed to 3) | ✅ (deep set) |
| The Problem (hook) | ✅ | ✅ |
| The Concept + diagram | ✅ | ✅ (extended) |
| Build It (from scratch) | ✅ **stays** (scaffold if budget tight) | ✅ (harder variant) |
| Use It (GTM weave) | ✅ **small RUNNABLE weave (15–30 lines)** | ✅ (full pipeline build) |
| Ship It (production) | ❌ → Loop 2 | ✅ **the core of Loop 2** |
| Exercises | 2 light | 5 graduated + capstone |
| Key Terms / Sources | ✅ | ✅ |

**The crew's correction (applied):** the from-scratch Build It stays — that's the non-tourist skill. But "Use It" is **not** a single named paragraph. It's a small runnable weave: 15–30 lines that fire what Build It just produced against one real GTM task. The GTM weave is the moat (the AI mechanism is commodity you can get free elsewhere); protecting it at runnable depth is the whole differentiator. If word budget forces a trade, scaffold the Build It before thinning the weave.

---

## 4. LOOP 1 — Fundamentals (build now)

### 4.1 Lean lesson spec
Write `shared/lean-lesson-spec.md`: target 1,200–1,800 words; required sections = Objectives (3) → Problem → Concept (+1 diagram) → Build It (runnable, from scratch) → Use It (15–30-line runnable GTM weave, AI mechanism named in the first sentence) → 2 exercises → Key Terms → Sources. No Ship It. This spec is the reconciliation target and the regen target.

### 4.2 Phase A — Reconcile (prerequisite to everything)
Build `scripts/reconcile_lessons.py`: walk all 510 `en.md`, classify by **content** (not manifest), rewrite manifest status truthfully into the 4 buckets above. Fixes "done = a file exists." Without it, any regen either no-ops or clobbers the 238 good lessons. **During this pass, harvest** each lesson's trimmed back-half (Ship It / heavy application) to `stages/02-lesson-injection/output/intermediate-seed/<zone>/<lesson>/` — nearly free here, and it preserves the entire Loop 2 corpus. **Commit the current 510 state first** (rollback = git).

### 4.3 Phase B — Complete to lean spec (small, gated)
- **Dispatcher fixes first:** raise `max_tokens` 8000→16000; **probe one lesson, assert `finish_reason=="stop"`** before any batch; log `finish_reason` per job; unify the three dispatchers onto one endpoint/config (touch-up currently diverges to bigmodel.cn with no token cap).
- **Two-tier write gate:** `structure-complete` (balanced fences, required headers, per-section min-chars) blocks the write; `ship-ready` (zero `[CITATION NEEDED]`, weave-names-mechanism check) blocks publish.
- **Work the buckets cheapest-first:** trim 238 → close 155 → complete 78 → regen 39. Write to sibling paths, validate, promote.

### 4.4 Phase C — Wire to site (bundle-at-build) + readability
- **`scripts/build_data.py`:** read the local lean lessons, emit `js/data.js` pointing at OUR content. **Bundle at build**, not runtime raw-GitHub fetch (kills CDN lag, the `master`-vs-`main` bug, push-coupling). Patch `lesson.js` `loadLesson` (hardcoded `/main/`, `phases/`-only regex).
- **Reader:** wire Mermaid.js (loading/parse-error/success states; keep a fallback until B's diagrams validate), Prism highlighting + copy button, reading-time. **Reading mode:** kill the grid inside `.lz-article`, lift off pure-black, demote red to links-only. VT323 display-only; headers/body in JetBrains Mono.
- **Open-Brain in the reader, honestly:** show the lesson's *static* `cards.json` count + an "Add to your deck" copy-command. Forgetting-curve animation goes on the **homepage**, never faked per-student in the reader (the static site cannot read the student's local `progress.json`).
- **Gate status the only honest way:** the site **advertises the recipe** (the artifact checklist) and **reads one boolean** — `intermediate_unlocked` in `progress.json` (already loaded). It never judges the gate; Helix does, in the terminal, and stamps the boolean. Site reflects a verdict the terminal earned.
- **Strip dead code:** auth.js, WP/Redis store adapters, HUD/XP/badge CSS, `api/` backend — and update `site-new/CLAUDE.md` golden rule 4 so the docs don't describe deleted code.

### 4.5 Phase D — First-run + the artifact-meter (crew's main correction)
- **First-run:** ship a pre-seeded `progress/progress.json` + pre-`init`'d FSRS deck in the clone; rewrite the empty-clone Helix greeting to hand over Zone-00-L01's URL instead of asking "what zone?"; add a homepage prerequisites strip (Claude Code Desktop, git, Python).
- **Meter, not wall.** Helix already scans the five artifact paths every session — it has the data, it just frames it as flavor. Reframe it as a **named, accruing "Loop 2 key" ledger** surfaced in every Zone-4+ greeting ("3 of 5 unlock conditions met — scrapers and handlers still open"). The wall never happens because the learner watches the meter fill the whole time.
- **The unlock beat (new deliverable).** Add a **fifth SESSION_START template** that fires *once*, on the scan transitioning incomplete→complete: leads with completion, names that Loop 2 is open, invites the first deep lesson. This is the payoff moment the plan was silently flipping a flag through.
- **Verdict in the terminal, soft-override for the owner.** Helix runs the check (existence + "does the scraper actually run," via exit code — not just file presence), writes `intermediate_unlocked: true` to `progress.json`. If incomplete, Helix *warns and offers "enter anyway"* rather than hard-blocking (the solo owner can't cheat himself; harden only if this ever becomes multi-student).

---

## 5. LOOP 2 — Intermediate (documented now, build later)

*Crew discipline applied: harvest the seed bank now (§4.2, free); defer the rest. This section is your brief for later, not work for now.*

### 5.1 The seed bank (the one Loop-2 thing done now)
Every "Ship It" section, full GTM pipeline, and hard exercise that Loop 1 trims is **harvested, not deleted** — written to `intermediate-seed/` during Phase A reconciliation. The 238 lean-ready lessons have complete deep halves; the truncated ones contribute partial seed. Loop 2 starts with a real corpus.

### 5.2 Intermediate lesson spec (write later)
Same 20 zones, the *hard* version: extended Concept, harder Build It, the **full** GTM pipeline build (complete CrewAI/Clay/Apollo systems, not a slice), production Ship It (deploy, scale, observability, cost), 5 graduated exercises + a zone capstone. ~3,000–4,000 words. The seed bank is its first draft.

### 5.3 — DECISIONS FOR YOU (capture while fresh — this is what you asked for)

These are the Loop 2 calls only you should make. I've put the crew's recommendation on each so future-you has a starting point, not a blank page.

1. **The artifact bar (what = "fundamentals complete").** Crew recommendation — **two-tier, mirroring the write gate:** *Structural (hard)* = all 20 zones' fundamentals lessons complete + FSRS deck active + `company.md`/`icp-definition.md` with no `{{PLACEHOLDERS}}` (honesty checks, can't be tourist-passed). *Functional (soft, warns)* = `signals/scrapers/` has a scraper that **runs** (exit 0) + `handlers/` imports clean. You define the exact components.

2. **Hard block vs soft-override.** Crew is unanimous: **soft-override while you're the only user** (warn, allow "enter anyway"); harden the day there's a second student. Your call — you were excited about the gate; this keeps the gate, just makes it humane for a solo learner. If you want it hard, say so.

3. **Per-zone vs global unlock.** Split decision in the crew. Design leaned per-zone (reward every zone, momentum). DX leaned global-unlock-once (preserves the single big payoff beat) with **per-zone visibility** on the map ("Zone 07 ✓ — scraper contributes to the key"). My lean: **global unlock, per-zone visibility** — one earned moment beats twenty small ones, and the meter keeps momentum without fragmenting the payoff.

4. **Does Loop 2 build ON the fundamentals artifact?** If yes, the gate is meaningful but Loop 2 exercises must tolerate a messy real artifact (ship a reference-implementation fallback for the from-scratch parts so a broken Loop-1 scraper doesn't brick a Loop-2 embeddings lesson). If no, the gate is theater. Recommendation: **yes for the GTM/Ship-It work** (your real company context is the point), **reference fallback for the pure-AI Build-It variants.**

### 5.4 Site expansion (later)
Same site. Each opened zone shows **two depth tiers** — Fundamentals (filled by lessons) and Intermediate (a dim "second floor" showing its artifact *recipe*, never a padlock — framed as earned depth, not a locked door). Progression stays advisory on the horizontal trail (20 zones, never 40 pins); depth is a vertical tier *inside* a zone — the two lock semantics never share a screen. Catalog gains a binary depth filter, defaults to Fundamentals, never interleaves tiers.

---

## 6. Sequencing & Ownership

```
A (reconcile + harvest seed bank) ──► B (complete to lean spec) ──► C (site wiring) ─┐
                                                                                       ├─► Loop 1 shippable
                              D (first-run + artifact-meter) ──────────────────────────┘
Loop 2: seed bank populated during A; spec + build deferred (§5 is its brief)
```

- **Conductor (me):** reconciliation + dispatcher architecture, `build_data.py` + reader wiring, the cut-line spec, the artifact-meter + unlock beat, sub-agent orders, commits.
- **GLM fleet (≤3 workers, gated):** the 117 completions/regens, behind the fixed dispatcher + two-tier gate.
- **Sub-agents:** site readability, dead-code strip, first-run seed.
- **Director:** the §5.3 decisions, when ready. Otherwise out of the loop.

**First three moves:** (1) commit current 510 state + `reconcile_lessons.py` (with seed-bank harvest); (2) dispatcher fixes + `finish_reason` probe; (3) write `lean-lesson-spec.md` and start the 238-lesson trim.

---

## 7. /autoplan Review Record

**Crew:** CEO · Design · DX (single independent voice each; Codex not yet installed). Three independent reviewers, no shared context, converged.

**Spine validated, no change:** reconcile-not-regenerate, the 77% reconciliation insight, lean spec, bundle-at-build, the honest Open-Brain reader, the first-run fixes.

### Auto-decided (strong cross-reviewer consensus → folded into plan)

| # | Decision | Principle | Source |
|---|----------|-----------|--------|
| 1 | **Artifact-gate → artifact-meter.** Visible accruing ledger from Zone 4; soft-override for the solo owner; Helix owns the verdict and stamps `intermediate_unlocked` into progress.json; site only reflects it. | Explicit / serve the real user | CEO + Design + DX (unanimous) |
| 2 | **Add the unlock payoff beat** — a 5th SESSION_START template firing once on completion. | Completeness | DX (Design + CEO concur) |
| 3 | **Gate check = "scraper runs" (exit code), not "file exists."** | Pragmatic | DX |
| 4 | **Cut line: Use It = small runnable weave (15–30 lines), not one paragraph.** Protect the moat. | Completeness | CEO |
| 5 | **Defer Loop 2 build; harvest seed bank now, document the rest.** Don't build two-product infra for an audience of one. | Bias to action / DRY | CEO |
| 6 | **Site never judges the gate** — advertises the recipe, reads one boolean. | Explicit | Design |
| 7 | **Two lock semantics never share a screen** — progression horizontal on map, depth vertical inside zone; 20 pins not 40; binary depth filter defaults to Fundamentals. | Explicit | Design |

### User challenge — flagged, NOT auto-decided
The CEO voice argued for **dropping the gate's blocking entirely** as cargo-culted SaaS mechanics. You were enthusiastic about the gate, so it stands as your direction — softened (meter + override per #1), not removed. Hard-block remains available if you want it (§5.3 #2). Your call governs.

### Open for the director (captured in §5.3)
The artifact bar, hard-vs-soft enforcement, per-zone-vs-global unlock, and whether Loop 2 builds on the artifact — documented with recommendations, deferred to you.
