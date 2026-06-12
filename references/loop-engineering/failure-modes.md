# Failure Mode Catalog

Real ways loops fail — and how good design mitigates them.

## Classification

| Severity | Meaning |
|----------|---------|
| **S1 — Annoying** | Wasted time/tokens, no user harm |
| **S2 — Harmful** | Wrong code merged, bad tickets, alert fatigue |
| **S3 — Critical** | Security, data loss, production incident |

## Infinite Fix Loop
**Symptom**: Same PR or CI job gets automated fix attempts 5+ times; never converges.
**Severity**: S2
**Mitigations**: Hard cap on attempts (e.g. 3) → escalate to human; separate verifier model; classify flakes in triage; record attempt count in state file.

## State Rot
**Symptom**: STATE.md references merged PRs, closed tickets, or stale branches.
**Severity**: S1 → S2
**Mitigations**: Prune closed/merged items every run; `Last run` timestamp + validate IDs against live API.

## Verifier Theater
**Symptom**: Verifier "approves" but tests fail in CI or review finds obvious bugs.
**Severity**: S2
**Mitigations**: Verifier must run test/lint commands and report output; different instructions: "find reasons to reject"; stronger model on verifier for unattended loops.

## Notification Fatigue
**Symptom**: Team mutes the bot; real escalations missed.
**Severity**: S1 → S2
**Mitigations**: Notify only when human decision required; digest mode for report-only loops.

## Token Burn
**Symptom**: Bill spikes; loop runs full sub-agent chains on empty triage.
**Severity**: S1
**Mitigations**: Cheaper triage-only pass first; spawn sub-agents only for actionable items; daily token budget → pause loop.

## Over-Reach (Wrong Scope)
**Symptom**: Loop refactors unrelated modules, touches denylisted paths.
**Severity**: S2 → S3
**Mitigations**: safety.md denylist enforced in skills; "Smallest possible diff" + verifier checks touched files.

## Comprehension Debt Spiral
**Symptom**: Velocity up, but no one can explain recent changes.
**Severity**: S2 (long-term)
**Mitigations**: Mandatory human review for non-trivial PRs; weekly "loop digest" read by owner.

## Cognitive Surrender
**Symptom**: "The loop handles it" — no opinions on correctness or design.
**Severity**: S2 (cultural)
**Mitigations**: Explicit human gates in every pattern; "Build it like someone who intends to stay the engineer."

## Parallel Collision
**Symptom**: Two sub-agents edit same files; merge conflicts; corrupted state.
**Severity**: S2
**Mitigations**: `isolation: worktree` for all code-editing sub-agents; lock in state: "PR #1234 — worktree in progress."

## Escalation Failure
**Symptom**: Loop stuck retrying; human never notified.
**Severity**: S2
**Mitigations**: Connector ping on escalation; alert if item in "waiting on human" section >24h.
