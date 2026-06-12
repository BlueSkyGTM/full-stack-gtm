# Core Concepts

Vocabulary for talking precisely about loop engineering. Use these terms in CONTEXT.md files, briefs, and stage contracts so agents don't re-derive them.

## Intent Debt

When skills (CONTEXT.md files, SKILL.md, project conventions) are missing or vague, agents re-derive intent from scratch on every run. Each re-derivation drifts slightly from the last. Over time the loop produces work that diverges from what the human actually wants — without the human noticing, because the output still looks plausible.

**The fix:** write skills before running loops. A skill is a bet that the up-front cost of writing it is lower than the cumulative cost of re-derivation.

## Comprehension Debt

The inverse of intent debt. When loops run autonomously and humans don't read the outputs carefully, velocity increases but understanding decreases. You end up with a codebase (or curriculum) you can't explain.

**The fix:** human gates, mandatory PR review for non-trivial changes, weekly loop digests. The loop should surface work for inspection, not make it easy to skip inspection.

## Cognitive Surrender

The failure mode where "the loop handles it" becomes a substitute for judgment. The human stops forming opinions about correctness, design, or direction — not because the loop is infallible, but because checking feels unnecessary.

Cognitive surrender is not a sudden event. It compounds gradually, one "looks fine" at a time.

**The fix:** explicit human gates in every pattern. Build the loop *like someone who intends to stay the engineer.*

## Orchestration Tax

The overhead of coordinating multiple agents — shared state collisions, worktree cleanup, escalation chains, token cost of running verifiers. Orchestration tax is real and non-zero; it must be weighed against the gains from parallelism and maker/checker separation.

A loop with 8 parallel subagents isn't 8× faster if 3 of them are blocked waiting on state locks and 2 are running verifier chains on work that didn't need verification.

**The fix:** start with the smallest loop that proves value. Add primitives incrementally.

## L1 / L2 / L3

Shorthand for the phased rollout of loop autonomy:

| Level | Behavior | When to move up |
|-------|----------|-----------------|
| L1 | Report-only — loop observes and surfaces, takes no action | After measuring triage accuracy |
| L2 | Assisted — loop acts on low-risk items, human reviews rest | After L2 verifier passes real scrutiny |
| L3 | Unattended — loop runs without watching | After Stage 10 validation / explicit trust grant |

Never skip L1. L3 is earned, not assumed.

## Maker / Checker Split

The loop pattern where the agent that produces work is structurally separated from the agent that verifies it. Separate instructions, often a stronger model on the verifier, always a default stance of REJECT.

This is the single most important pattern for reliable autonomous loops. Without it, verification is theater.

## Kill Switch

Explicit criteria for pausing or stopping a loop. Every production loop must document: what triggers a pause, who can restart it, and what state to inspect before restarting.

A loop without a kill switch is a loop you can't safely stop in an emergency.
