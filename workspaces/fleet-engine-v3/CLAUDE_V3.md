# CLAUDE_V3.md — Fleet Engine V3 Design Brief

## What V3 Is

A self-improving GTM pipeline. The core agent triad (Ahab, Nemo, Neptune) is unchanged.
V3 adds a feedback loop: reply outcomes from HubSpot flow back into the RAG store,
and a fourth agent — the Polish pass — uses that signal to improve Outreach Bites
before delivery. The pipeline learns which Bites get replies and writes better ones over time.

Do not build V3 until Phase 2 of V2 is confirmed working.

---

## The Problem V3 Solves

V2's RAG store compounds enrichment and discovery signal across runs, but it has
no outcome data. Neptune gets better at varying Bite structure (no repeated openings,
villains, or CTAs) but has no signal for what actually converts. It optimizes for
novelty, not effectiveness.

V3 closes the loop: outcomes come back in, the store learns what works by friction
type and contact role, and the Polish pass applies that learning before every delivery.

---

## New Component: Reply Feedback Loop

### What it does
After a sequence runs in HubSpot, outcomes (replied, opened, bounced, booked)
are written back to the RAG store as `outcome` entries. Each entry links back
to the `bite` entry it corresponds to via `bite_id`.

### Entry shape
```json
{
  "id": "uuid",
  "type": "outcome",
  "run_id": "string",
  "timestamp": "ISO string",
  "text": "string — the Bite that was sent",
  "embedding": [768 floats],
  "metadata": {
    "bite_id": "string — links to the bite entry",
    "company_name": "string",
    "friction_type": "string",
    "contact_title": "string",
    "service_intent": "string",
    "outcome": "replied | opened | bounced | booked",
    "outreach_bite": "string"
  }
}
```

### How outcomes get into the store
Two options — pick one at V3 build time:

**Option A — HubSpot webhook (real-time)**
HubSpot fires a webhook on contact activity events (reply, open, booking).
A lightweight endpoint (Cloud Run or Cloud Functions) receives it, looks up
the matching `bite` entry in the RAG store by `company_name` + `contact_email`,
and writes an `outcome` entry.

**Option B — Manual CSV import (simpler, Phase 3 start)**
Export HubSpot sequence outcome report as CSV. Run `node scripts/import-outcomes.js`
which reads the CSV, matches rows to RAG store `bite` entries, and bulk-upserts
`outcome` entries. No webhook infrastructure needed. Good enough for first 500 leads.

Start with Option B. Graduate to Option A when volume justifies it.

---

## New Agent: Polish

### Where it lives
`agents/polish.js` — runs between Neptune and `deliver()` in `run.js`.

### What it does
For each finished Neptune lead, Polish queries the RAG store for `outcome` entries
with matching `friction_type` and `contact_title`. It retrieves the top-K highest-converting
Bites (outcome = "replied" or "booked") and the top-K lowest-converting Bites
(outcome = "bounced" or no open). It passes both sets to Gemini 2.5 Flash with a
targeted rewrite prompt: keep the structure, improve the specific elements that
distinguish the high-converting examples from the low-converting ones.

Polish does not rewrite every Bite. It skips leads where:
- Fewer than 5 `outcome` entries exist for that friction_type (not enough signal)
- The matching outcomes are split evenly (no clear pattern yet)

In those cases it passes the Neptune Bite through unchanged.

### Model
Gemini 2.5 Flash — this is a polish pass, not deep reasoning. Speed matters.
Polish runs after Neptune has already done the synthesis work.

### System prompt shape (to be written at V3 build time)
```
[Task]: BITE REFINEMENT based on outcome signal.
[Input]: A Neptune Outreach Bite + high-converting reference Bites + low-converting reference Bites.
[Mission]: Identify the specific structural or linguistic patterns that distinguish
the high-converting examples. Apply those patterns to the input Bite without
changing the friction type, villain, or contact frame.
[Constraint]: Do not change the core observation. Only refine the expression.
[Output]: One improved Bite. Same length. Raw string only.
```

---

## Updated Pipeline Shape

```
Ahab → Nemo → Neptune → Polish → deliver() → Clay → HubSpot
                                                         │
                              RAG store ◄────────────────┘
                              (outcome entries written back
                               via webhook or CSV import)
```

---

## RAG Store Evolution

| Phase | Entry types | What compounds |
|---|---|---|
| V2 Phase 1 | discovery, enrichment, bite | Dedup, friction calibration, Bite variety |
| V2 Phase 2 | + (Clay/HubSpot confirmed) | Delivery routing confirmed |
| V3 | + outcome | Bite effectiveness by friction type + contact role |

By run 20 with outcome data, Polish has a real signal corpus. By run 50, the
pipeline is measurably improving reply rates without any prompt changes.

---

## What V3 Does Not Change

- Ahab, Nemo, Neptune system prompts — unchanged, agent preservation rule still applies
- GCS handoff architecture — unchanged
- Clay delivery architecture — unchanged
- HubSpot as system of record — unchanged
- `SKIP_RAG`, `DRY_RUN`, `AHAB_ONLY`, `NEMO_ONLY`, `DELIVER` flags — all still work
- New flag: `SKIP_POLISH=true` — bypasses Polish pass, passes Neptune output directly to deliver()

---

## Build Order for V3

1. `scripts/import-outcomes.js` — CSV → RAG store `outcome` entries (Option B)
2. `agents/polish.js` — reads outcomes, rewrites Bites where signal exists
3. Update `run.js` — add Polish between Neptune and deliver(), add `SKIP_POLISH` flag
4. Update `memory/rag.js` — add `outcome` entry type, `outcomeText()`, `formatOutcomeContext()`
5. Update `STATE.md` — track reply rates per run alongside lead counts

Gate to start V3: 50+ outcome entries in RAG store (one full sequence cycle completed).
