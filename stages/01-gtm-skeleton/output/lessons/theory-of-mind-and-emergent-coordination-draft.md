# Theory of Mind and Emergent Coordination

## Hook

Language models don't have beliefs—but they can simulate reasoning about the beliefs of others. This beat introduces the Sally-Anne false-belief task as it applies to LLMs, and asks whether modeling other agents' mental states is mechanistically real or pattern-matched. Sets up the central tension: useful behavior vs. credited capability.

## Concept

Covers three mechanisms in order: (1) Theory of Mind (ToM) as the ability to represent another agent's knowledge state as distinct from one's own, tested via false-belief, unexpected contents, and second-order belief tasks; (2) emergent coordination, where multiple agents achieve aligned behavior without a central controller, driven by shared context windows or observation of each other's outputs; (3) the skeptical reading—ToM-like outputs from LLMs may be shallow pattern completion on training data that embedded human narrative structures, not evidence of an internal world model. Distinguishes first-order ("she thinks X") from second-order ("he thinks she thinks X") reasoning.

## Detect

Practitioner runs a battery of false-belief prompts through an LLM and measures accuracy across first-order, second-order, and decoy (true-belief control) conditions. Also tests coordination: two LLM instances share a conversation log and must converge on a joint decision without explicit negotiation protocol. Observable output: accuracy scores per condition, convergence round count, and failure mode classification.

**Exercise hooks:**
- *Easy:* Run a single Sally-Anne variant through an LLM and print the model's answer alongside the correct answer.
- *Medium:* Build a false-belief battery (5 variants) and compute accuracy per order of reasoning.
- *Hard:* Implement two LLM agents that must coordinate on a resource allocation task using only shared transcript access—measure rounds to convergence and detect deadlocks.

## Use It

ToM-like capability maps to any GTM task that requires modeling what a prospect knows, believes, or cares about. The specific GTM cluster: **Zone 3 (Outreach) — personalization and sequencing**. When an agent crafts outreach, it must represent the buyer's current state (awareness level, objections, competitive exposure) as distinct from the seller's knowledge. Emergent coordination maps to multi-agent GTM workflows where enrichment agents, scoring agents, and outreach agents operate on shared context without a rigid orchestrator—directly relevant to Clay's waterfall model and multi-step enrichment chains.

**Exercise hooks:**
- *Easy:* Prompt an LLM to generate a cold email that accounts for a prospect who has never heard of your category (explicit false-belief condition: "the prospect does not know what X is").
- *Medium:* Build a two-prompt pipeline: prompt one models the buyer's current belief state given firmographic signals; prompt two generates outreach calibrated to that belief state.
- *Hard:* Implement a minimal multi-agent enrichment chain where Agent A enriches (industry, tech stack), writes findings to shared context, and Agent B generates outreach using only that context—no direct handoff instructions.

## Ship It

Production considerations: ToM prompting is unreliable at second-order reasoning in current models—never trust it for compliance-critical decisions. Always validate coordination outputs against a deterministic fallback. Covers prompt templates for belief-state modeling, shared-context patterns for multi-agent coordination, logging for debugging emergent misalignment, and a checklist for when ToM-based personalization is safe to ship versus when to use explicit rules.

**Exercise hooks:**
- *Medium:* Wrap the two-agent coordination from "Detect" in a logging layer that records each agent's internal "belief" about the other agent's state per round—output a coordination trace.
- *Hard:* Build a guardrail that detects when an agent's ToM reasoning has drifted (output no longer consistent with stated prospect beliefs) and falls back to a rule-based template.

## Evaluate

**Learning Objectives:**
1. Implement a false-belief test battery and classify model outputs as correct, pattern-matched, or confabulated.
2. Compare first-order and second-order ToM performance in an LLM using controlled prompt variants.
3. Build a minimal multi-agent coordination loop using shared context and measure convergence behavior.
4. Diagnose coordination failures (deadlock, divergence, cascade) from agent transcripts.
5. Determine when ToM-based personalization is appropriate for a GTM workflow versus when explicit rules are required.

**Quiz hooks (not full questions):**
- Identify whether a given LLM output demonstrates true false-belief reasoning or a true-belief shortcut.
- Classify a coordination failure trace as deadlock, divergence, or cascade.
- Given a GTM scenario (e.g., sequencing to a prospect who has evaluated a competitor), determine the appropriate order of ToM reasoning needed.
- Evaluate whether emergent coordination or explicit orchestration is the correct pattern for a described multi-agent GTM workflow.

---

**GTM Redirect Rules for this lesson:**
- Primary cluster: **Zone 3 — Outreach sequencing and personalization** (ToM enables modeling buyer belief state; emergent coordination enables multi-agent enrichment-to-outreach chains)
- Secondary cluster: **Zone 2 — Multi-step enrichment** (emergent coordination pattern applies to waterfall-style enrichment where agents build on shared context)
- Foundational note: Second-order ToM reasoning in LLMs is not reliable enough for production use—this is documented behavior and the lesson teaches detection of failure modes, not reliance on the capability