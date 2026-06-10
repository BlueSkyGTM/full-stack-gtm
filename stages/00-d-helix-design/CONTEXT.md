<!-- Agent: Claude Code -->
# 00-d: Helix Design

Design the FSRS integration, copy-paste flag format (LOCKED here), faculty persona system, and student state options.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Curriculum archaeology | `../00-a-curriculum-archaeology/output/` | exercise-format-spec, lesson-format-spec | What Helix must parse |
| Agent architecture | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | How Helix fits in the agent pattern |
| FSRS reference | `references/fsrs-algorithm.md` | Full file | Algorithm spec and test assets |
| Professor Synapse | `{{SYNAPSE_REPO_URL}}` | Existing implementation | Fork base |

## Process

1. Evaluate FSRS for this use case: text-based curriculum with CLI exercises
2. Define FSRS integration spec: card format, scheduling parameters, review intervals, correct-response definition
3. Design copy-paste flag format: exact string Helix parses from student CLI output — **LOCKED after this stage**
4. Design faculty persona system: trigger conditions, persona types (GTM vs AI engineering topics), voice rules
5. Evaluate student state options: what data must persist, mechanism candidates, evaluation criteria
6. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| Flag format is exact | copy-paste-flag-format.md specifies the exact string, not a description |
| FSRS params are concrete | fsrs-integration-spec.md has actual parameter values, not ranges |
| Stage 07 criteria are decision-ready | student-state-options.md names criteria beyond "persistent, cheap, reliable" |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `fsrs-integration-spec.md` | `output/` | Card schema, scheduling params, correct-response definition |
| `copy-paste-flag-format.md` | `output/` | LOCKED — exact string format Helix parses |
| `faculty-persona-spec.md` | `output/` | Persona types, trigger conditions, voice rules |
| `student-state-options.md` | `output/` | Mechanism candidates and evaluation criteria |
