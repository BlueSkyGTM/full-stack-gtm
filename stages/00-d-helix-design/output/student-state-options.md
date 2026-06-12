# Student State — Architecture Decision
<!-- Stage 00-d output | 2026-06-12 (revised) -->
<!-- STATUS: DECIDED — repo-as-save-file. See Stage 07 CONTEXT.md for implementation spec. -->

## The Model: The Repo Is the Save File

Student state lives in `progress/progress.json` inside the student's **mission command fork** (the Albatross). Helix writes to it. The student commits it. Their fork is the cartridge — progress travels with the repo, not with the site.

This is not an option. It is the architecture. The site auth backend is gutted.

---

## What This Means for the Site

The site (`site-new/`) is **read-only content delivery**. It shows lessons. It does not track who has read them. It does not store quiz answers. It does not know who is logged in.

**Changes from current state:**
- `api/auth.js` — REMOVE. GitHub OAuth is not needed.
- `api/progress.js` — REMOVE. Vercel KV progress sync is not needed.
- `api/_lib/auth.js` — REMOVE.
- `site-new/js/auth.js` — REMOVE or gut to a no-op. No cookie parsing, no adapter swap.
- `site-new/js/store.js` — SIMPLIFY. Remove `vercelAdapter`. `localAdapter` may stay for the lesson visit log if desired — but it is cosmetic, not canonical. The canonical save file is the repo.
- `js/data.js` — UNCHANGED. Content delivery still needs the PHASES data.

**What replaces the auth/progress layer:**
Nothing on the site. Progress is tracked in the mission command repo. The student's state is their repo. If they want to see their progress, they look at `progress/progress.json` or ask Helix.

---

## What Must Persist (and Where)

| State Layer | Data | Location | Lifecycle |
|-------------|------|----------|-----------|
| **Artifact gates** | Which mission-command components exist and are configured | Filesystem — Helix reads the repo directly | Permanent while the file exists |
| **Quiz recall (FSRS)** | Per-card stability, difficulty, due date, lapses | `progress/progress.json → fsrs: {}` | Never cleared automatically |
| **Lesson visits** | Optional visit log for student reference | `progress/progress.json → lessons: {}` | May be cleared on student request |
| **Business configuration** | `context/company.md`, ICP, signals, playbooks | Mission-command repo context files | Updated by student via weekly-update |
| **Rename** | Student's name for their command center + Helix | `CLAUDE.md` header + `STATE.md` | Written once when operator mode is earned |

---

## Gate Logic (How Helix Knows What's Unlocked)

Gates are **artifact-based**. Helix checks filesystem state, not a certificate or a checkbox:

```
context/company.md — no {{PLACEHOLDERS}}?          → Stage 01 cleared
signals/scrapers/ — at least one scraper exists?   → Stage 06 cleared
handlers/research-handler/ — exists + valid?       → Stage 08 cleared
progress/progress.json — quiz scores present?      → lesson-level recall gates
```

Helix does not say "Stage 06 is locked." It says "your research handler is trying to read from `signals/processors/` but that directory is empty — looks like your signal processor isn't built yet." It redirects to the first missing piece.

**Anti-cheat is structural.** Cloning someone else's completed repo gives you their business, their ICP, their scrapers targeting their signals. There is nothing to gain. Progress is only meaningful in the context of the business the student configured.

---

## The Naming Mechanic (Operator Mode)

When all gates clear (Stage 10 validation complete):

1. Student can rename their command center. The name propagates through `CLAUDE.md`, `STATE.md` header, and Helix's self-references.
2. Student can rename Helix. Same propagation.

The gate is hard. Nobody earns operator mode by building the course — they earn it by running it. The course author does not get operator mode until they complete the validation run on their own instance.

---

## What This Means for Stage 07

Stage 07 implements:
1. Gut the site auth backend (remove the three API files, simplify store.js)
2. Implement gate-check-spec: artifact existence checks, placeholder detection, quiz score thresholds
3. Implement Helix redirect logic: missing component → identify first gap → surface to student
4. Write editor-mode bypass: `.editor-mode` file → all gate checks return `cleared: true` (testing harness, not operator mode)
5. Write rename mechanic: `/rename <name>` Helix command, available only when `progress.json` shows all Stage 10 gates cleared

See `stages/07-student-state/CONTEXT.md` for the full implementation spec.

---

## What This Does NOT Change

- FSRS-integration-spec.md — algorithm and card format unchanged; only storage location moves to mission command repo
- Copy-paste flag format — unchanged; Helix still parses `BLUESKYGTM_CHECK: OK` from student CLI output
- Helix ramp schedule — unchanged; capability levels still unlock per zone

## HUMAN GATE: Resolved

Decision: **Repo-as-save-file (mission-command fork).**
Site auth backend: **gutted.**
Decided by: repo architecture (Stage 07 CONTEXT.md pre-specified this model).
