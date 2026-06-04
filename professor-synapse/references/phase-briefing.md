# Phase Briefing — How to Read BATCH.md

This file explains the BATCH.md structure that Synapse reads when loading a phase.
Each phase directory contains a BATCH.md that is the domain briefing for that phase.

---

## Where to Find BATCH.md

```
phases/NN-phase-slug/BATCH.md
```

Example: `phases/07-transformers/BATCH.md`

If BATCH.md doesn't exist in a phase directory, fall back to reading the first lesson's
`docs/en.md` to understand the domain vocabulary.

---

## BATCH.md Sections (expected)

A well-formed BATCH.md contains:

### Domain Vocabulary
Key terms specific to this phase. These are the concepts Synapse must know to teach
fluently — not definitions to read aloud, but vocabulary to internalize before engaging.

### Lesson List
The ordered list of lessons in this phase with their slugs and brief descriptions.
Use this to understand the phase arc: what's introduced early vs. built on later.

### Build It / Use It Split
Which lessons require the student to implement from scratch vs. use a library.
This shapes teaching depth — "build it" lessons warrant more Socratic pressure on
the implementation details. "Use it" lessons warrant more focus on intuition and limits.

### Common Misconceptions
If the BATCH.md includes a misconceptions section, prioritize these in teaching.
These are the gaps most students hit. Watch for them before the student shows them.

### Prerequisites
Which earlier phases this phase depends on. If a student gaps on a concept here,
check the referenced phase first before explaining directly.

---

## Phase Loading Protocol

When Synapse enters a new phase (student is in a phase with no recent observation):

1. Read BATCH.md — internalize vocabulary and lesson arc
2. Run `python3 scripts/query_graph.py query "[phase domain]"` to see what concepts connect
3. Check INDEX.md — is there a faculty specialist for this phase?
   - YES: load the specialist, they have phase-specific patterns
   - NO: teach from BATCH.md + docs/en.md, note that domain-researcher should build a specialist after the session
4. Read the current lesson's docs/en.md — know it well enough to guide without re-explaining it

---

## Lesson Directory Structure

Each lesson within a phase follows this shape:

```
phases/NN-phase-slug/
  MM-lesson-slug/
    docs/en.md          ← primary lesson document (READ THIS)
    code/               ← implementation reference
    quiz.json           ← comprehension questions
    outputs/            ← expected outputs
```

The `docs/en.md` is the lesson contract. Everything Synapse points the student toward
should be findable in that file. If a concept is not in docs/en.md, it is either a
prerequisite (handle with direct explanation or bridge) or outside scope.

---

## What to Do When BATCH.md is Missing

Some phases may not have a BATCH.md. If so:

1. Read the first three lessons' `docs/en.md` to understand the domain vocabulary
2. Note vocabulary gaps in synapse-observations.jsonl as `type: "gap"` with `concept: "BATCH.md absent"`
3. Proceed with teaching from the lesson files directly
4. Signal to domain-researcher that a BATCH.md should be generated for this phase

Do not block teaching on the absence of BATCH.md.
