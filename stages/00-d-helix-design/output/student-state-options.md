# Student State Options
<!-- Stage 00-d output | 2026-06-12 -->
<!-- HUMAN GATE: Review this file and record the mechanism decision before running 00-e-full. -->
<!-- Decision field at bottom of this file — fill it in before proceeding. -->

## What Must Persist

Student state has two distinct layers that must be stored separately:

| Layer | Data | Lifecycle |
|-------|------|-----------|
| **Progress** | `done` (lesson completions), `days` (streak data), `updatedAt` | Reset on explicit student request. NOT cleared on logout. |
| **FSRS state** | Per-card stability, difficulty, due date, review count, lapses | NEVER cleared. Long-term memory. Persists across logout, device change, and explicit progress reset. |

These are separate namespaces in the schema (see `fsrs-integration-spec.md` for the full schema shape). Any mechanism that clears both layers together is architecturally incorrect.

## Mechanism Candidates

### Option A: Vercel KV (current path — extend what exists)

**How it works:** The existing `/api/progress` endpoint already reads and writes to Vercel KV when the student is authenticated. FSRS state is a new field in the same KV record. Unauthenticated students use `localStorage` as today.

**What changes:**
- Extend the KV schema to add `fsrs: {}` field
- Ensure `adapter.clear()` on logout zeros `done` but not `fsrs`
- No new infrastructure

**Evaluation:**
| Criterion | Score |
|-----------|-------|
| Persistent across devices | ✓ (authenticated only) |
| Works unauthenticated | Partial (localStorage, lost on browser clear) |
| Cost | Vercel KV free tier: 30MB, 3M ops/month — sufficient for beta |
| Implementation complexity | Low — additive schema extension |
| FSRS separation from progress | Requires explicit clear() guard (auth-audit.md identified this risk) |
| Dependency risk | Tied to Vercel platform |

**Stage 07 work required:** env guard for `GITHUB_CLIENT_ID`/`SITE_URL`, non-breaking schema extension, `adapter.clear()` guard.

---

### Option B: Local FSRS + cloud progress sync

**How it works:** FSRS state lives exclusively in `localStorage` (never uploaded). Progress (`done`, `days`) syncs to Vercel KV for authenticated users. Two separate adapters for two separate state layers.

**What changes:**
- New `fsrsAdapter` in `store.js` that always reads/writes `localStorage` regardless of auth state
- Existing `vercelAdapter` handles `done`/`days` only
- FSRS state never leaves the browser

**Evaluation:**
| Criterion | Score |
|-----------|-------|
| Persistent across devices | ✗ (FSRS state is per-browser) |
| Works unauthenticated | ✓ (localStorage works for both layers) |
| Cost | No additional Vercel KV ops for FSRS |
| Implementation complexity | Medium — two adapter types, explicit routing |
| FSRS separation from progress | Strong — structurally separate by design |
| Dependency risk | Low — FSRS doesn't require backend |

**Tradeoff:** FSRS is the long-term memory layer. Students who change devices lose their review history. For a solo learner, this may be acceptable in beta.

---

### Option C: gbrain student brain (Helix open-brain)

**How it works:** Each student gets a gbrain brain instance. FSRS state, progress, and conversation history all live in the student's pglite database. The student carries their brain; Helix reads from it directly via MCP.

**What changes:**
- Student brain provisioning on first login (not built yet — new infrastructure)
- Helix reads `next_card` and `review_summary` from gbrain rather than from a JSON field
- Progress sync via gbrain rather than Vercel KV

**Evaluation:**
| Criterion | Score |
|-----------|-------|
| Persistent across devices | Conditional — requires gbrain sync setup |
| Works unauthenticated | ✗ (gbrain requires setup) |
| Cost | gbrain is local (pglite) — no backend cost, but setup friction |
| Implementation complexity | High — new student provisioning layer, MCP integration |
| FSRS separation from progress | Excellent — gbrain is designed for this |
| Dependency risk | Medium — gbrain is a new dependency |

**Note:** Option C is the Helix open-brain vision from the course identity doc. It is not a beta option — it is the target architecture. The question is when, not whether.

---

## Evaluation Criteria

When making the mechanism decision, weight these criteria:

1. **FSRS state survives logout** — this is non-negotiable. See auth-audit.md, Stage 07 implication 4.
2. **Works for a solo learner in beta** — the first student is the course author. Infrastructure complexity is a real cost.
3. **Path to multi-device FSRS sync** — the mechanism should not require a rewrite to add multi-device support later.
4. **Non-breaking from current auth flow** — the adapter pattern must not be replaced, only extended.
5. **Vercel KV schema extension is additive** — new fields only, no migrations.

---

## Decision (HUMAN GATE — fill this in before running 00-e-full)

```
MECHANISM DECISION: [ Option A / Option B / Option C ]

Decided by: ________________
Date: ________________

Rationale (one sentence):
________________

Stage 07 scope change (if Option B or C): ________________
```

**Do not run 00-e-full until this section is filled in.**

00-e-full's `helix-voice.md` includes a section on how Helix communicates state to students ("Your review streak is X" vs "Your cards are synced across devices"). The voice copy changes depending on which mechanism is selected.
