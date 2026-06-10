# Student Profile — How to Read and Write Progress Data

This file explains the progress data schema and Synapse's write conventions.

---

## Reading aifs-progress.json (read-only)

Path: `progress/aifs-progress.json`

```json
{
  "lessons": {
    "07:3": "completed",
    "07:4": "in_progress"
  },
  "focus": {
    "currentLesson": "07:4",
    "mode": "study",
    "note": "working through multi-head attention"
  }
}
```

The `lessons` map uses `phaseId:lessonIndex` as the key (e.g. `"07:3"` = Phase 07, 4th lesson).
`"in_progress"` means the student has started but not completed the lesson.
`"completed"` means the student marked it done (or quiz passed, depending on site config).

`focus.currentLesson` is the lesson the student was on last. Start here if no other signal exists.

Do NOT write to this file. It is owned by the site's store.js.

---

## Reading learning-profile.json (read-only)

Path: `progress/learning-profile.json`

```json
{
  "preferences": {
    "pace": "thorough | fast | null",
    "depth": "deep | surface | null",
    "feedback": "direct | gentle | null",
    "sessionLength": "short | long | null"
  },
  "strengths": ["matrix ops", "backprop intuition"],
  "weaknesses": ["probability theory", "GPU memory management"],
  "doubts": ["am I ready for transformers?"],
  "avoid": ["lengthy theory before code"]
}
```

Use this to calibrate tone and depth. If `preferences.pace = "fast"`, be direct.
If `strengths` lists a concept that the current lesson builds on, reference it.
If `weaknesses` lists a concept the current lesson requires, watch for it.

Do NOT write to this file. The student updates it manually.

---

## Writing synapse-observations.jsonl (write here)

Path: `progress/synapse-observations.jsonl`

Synapse writes observations during sessions — immediately when a pattern is spotted,
not at session end (session end is not a reliable hook).

Entry shapes:

```jsonl
{"ts": 1717000000, "type": "pattern", "concept": "attention masking", "phase": "07", "note": "student confuses causal vs padding masks consistently"}
{"ts": 1717000001, "type": "strength", "concept": "matrix ops", "phase": "07", "note": "student built attention from raw numpy without prompting"}
{"ts": 1717000002, "type": "gap", "concept": "softmax stability", "phase": "07", "note": "student unaware of numerical stability issue with naive softmax"}
{"ts": 1717000003, "type": "bridge", "from": "backprop", "to": "transformer training", "phase": "07", "note": "connection clicked when framed as 'gradient still flows, just through attention weights'"}
```

Types:
- `pattern` — a consistent behavior or misconception to watch for in future sessions
- `strength` — something the student does well (use to calibrate difficulty)
- `gap` — a specific missing piece of knowledge in the current phase
- `bridge` — a cross-phase connection that worked (useful reference for future sessions)

The student merges these into `learning-profile.json` when they choose.
This keeps Synapse's writes auditable and the student in control.

---

## Session Start Decision Tree

```
1. Read aifs-progress.json
   → focus.currentLesson set?
     YES → load that lesson's docs/en.md + phase BATCH.md
     NO  → ask: "Where do you want to start today?"

2. Read learning-profile.json
   → any weaknesses that overlap with current phase? Note them.
   → any preferences set? Calibrate accordingly.

3. Read synapse-observations.jsonl (if exists)
   → any recent patterns for current phase? Reference them.
   → last session's gaps still unresolved? Start there.

4. Check agents/INDEX.md
   → specialist exists for current phase? Load them.
   → no specialist? Teach from BATCH.md, domain-researcher creates one after.
```
