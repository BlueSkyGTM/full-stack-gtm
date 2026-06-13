# Bounded Self-Improvement Designs

## Hook

Unbounded self-improvement is alignment in miniature — a system that rewrites its own prompts or tool selection without constraints will optimize for whatever metric you give it, including gaming that metric. This lesson covers the architecture of improvement loops with hard bounds: invariants that cannot be violated regardless of score.

## Concept

Three components form the mechanism: an **improvement operator** (mutates the artifact), a **scoring function** (measures quality), and a **gatekeeper** (enforces invariants). The improvement operator proposes changes. The gatekeeper rejects any mutation that violates defined bounds — syntactic constraints, semantic distance from an anchor, or behavioral whitelists. Score only matters if the gatekeeper passes the variant. This is not reinforcement learning; it is generate-and-test with enforced constraints.

## Demo

Build a minimal self-improving prompt optimizer in Python. The system takes a task prompt, runs it against a test case, scores the output, mutates the prompt using an LLM call, re-runs, and accepts the mutation only if: (a) the score improves AND (b) the mutated prompt passes a length bound and a keyword-inclusion check. Print score trajectory, accepted mutations, and rejected mutations with rejection reasons. All code runs in terminal with observable output.

## Use It

Apply bounded self-improvement to GTM outbound message optimization. The improvement operator rewrites cold email subject lines. The scoring function estimates open rate from heuristics or historical data. The gatekeeper rejects any variant whose embedding distance from approved brand-voice examples exceeds a threshold. This maps to the **Message Testing and Optimization** cluster in the GTM topic map — automated variant generation constrained to brand safety bounds. [CITATION NEEDED — concept: self-improving GTM message optimization loop]

## Ship It

Ship a bounded self-improvement loop that tunes a real workflow artifact: cold email sequence, lead scoring weights, or ICP description. The practitioner defines: the artifact to improve, the scoring function (must be measurable), and at least two invariant bounds (hard constraints). The system runs N improvement cycles, logs all proposals with accept/reject decisions, and outputs the final artifact alongside the improvement history.

## Evaluate

**Easy:** Modify the gatekeeper in the demo to add one additional bound and show it rejecting a mutation that would have passed previously. **Medium:** Design a scoring function for a GTM artifact (e.g., lead scoring accuracy) and wire it into the improvement loop with appropriate bounds. **Hard:** Implement a two-level improvement system where the improvement operator itself is evaluated — if it produces three consecutive rejected proposals, fall back to a simpler mutation strategy. Document the failure modes observed when bounds are too tight (stagnation) versus too loose (drift from intended behavior).