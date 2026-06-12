<!-- Agent: Claude Code -->
# 00-d: Helix Design

Design the FSRS integration, copy-paste flag format (LOCKED here), faculty persona system, and student state options.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Curriculum archaeology | `../00-a-curriculum-archaeology/output/` | exercise-format-spec, lesson-format-spec | What Helix must parse |
| Agent architecture | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | How Helix fits in the agent pattern |
| FSRS reference | `../../phases/19-capstone-projects/17-personal-ai-tutor/docs/en.md` | FSRS definition, BKT section | Algorithm spec — use FSRS-5 |
| Helix architecture | `../../vault/helix-architecture.md` | Full file | Governed-maze design, system prompt layers, decision tree — Helix is built from this spec, not forked from any existing agent |

## Capstone Direction

Helix uses FSRS-5 for quiz card recall scheduling. The Phase 19/20 capstone extends or ports the algorithm: students either tune FSRS parameters for GTM-specific retention curves, or apply the stability/decay model to signal resurfacing in the GTM Starter Kit. Both directions are valid. 00-d's spec locks the base parameters; the capstone layer builds on top.

## Process

1. Evaluate FSRS-5 for this use case: text-based curriculum with CLI exercises. Confirm compatibility with quiz card layer (atomic Q&A cards with Again/Hard/Good/Easy ratings). Note: copy-paste exercises are a separate layer — FSRS does not apply to them.
2. Define FSRS integration spec: card format, scheduling parameters (desired retention rate, stability thresholds, learning/relearning steps), review intervals, correct-response definition
3. Design copy-paste flag format: exact string Helix parses from student CLI output — **LOCKED after this stage**
4. Design faculty persona system: trigger conditions, persona types (GTM vs AI engineering topics), voice rules
5. Evaluate student state options: what data must persist, mechanism candidates, evaluation criteria
6. **FSRS validation (required before locking):** Write one worked example quiz card — a real question from the curriculum, with an Again/Hard/Good/Easy response and the resulting FSRS-5 interval output. Confirm the desired retention rate and stability parameters produce a schedule that makes sense for a 3-week learner. If the intervals are clearly wrong (e.g. first review in 40 days), revise the spec before locking.
7. Run audit checks

## Human Gate — REQUIRED before 00-e-full

After this stage completes, a human must review `student-state-options.md` and make the architecture decision. Do NOT run 00-e-full until this decision is recorded. 00-e-full's helix-voice.md depends on which student state mechanism is selected.

## Re-run / Relock Safety

`copy-paste-flag-format.md` and `fsrs-integration-spec.md` are **LOCKED** after this stage. If you rerun 00-d and produce new versions of these files:
1. Every Stage 05 Helix exercise that has already been built must be rebuilt against the new spec.
2. Confirm explicitly before overwriting: "I am intentionally relocking — I understand all downstream Stage 05 exercises must be rebuilt."
The audit check below enforces this by requiring an explicit confirmation line in the spec files.

## Audit

| Check | Pass Condition |
|-------|---------------|
| Flag format is exact | copy-paste-flag-format.md specifies the exact string, not a description |
| FSRS params are concrete | fsrs-integration-spec.md has actual parameter values, not ranges |
| FSRS params validated | A worked example quiz card with computed intervals is present in fsrs-integration-spec.md |
| Stage 07 criteria are decision-ready | student-state-options.md names criteria beyond "persistent, cheap, reliable" |
| Human gate acknowledged | Human has reviewed student-state-options.md and recorded the mechanism decision before 00-e-full runs |
| Helix ramp schedule produced | helix-ramp-schedule.md exists and specifies per-phase capability level for at least phases 01–10 |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `fsrs-integration-spec.md` | `output/` | Card schema, scheduling params, correct-response definition |
| `copy-paste-flag-format.md` | `output/` | LOCKED — exact string format Helix parses |
| `faculty-persona-spec.md` | `output/` | Persona types, trigger conditions, voice rules |
| `student-state-options.md` | `output/` | Mechanism candidates and evaluation criteria |
| `helix-ramp-schedule.md` | `output/` | Per-phase Helix capability level: what Helix can do in each phase (tutor only / tutor + quiz scoring / tutor + light context-read / full context-read / Revelation 1 trigger). Required by Stage 06 to implement the gradual introduction. |
