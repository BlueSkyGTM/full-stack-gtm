# Dual-Use Risk — Cyber, Bio, Chem, Nuclear Uplift

## Hook
A single LLM interaction can both harden a cloud intrusion detection signature and refine an exploit chain that bypasses it. This lesson treats dual-use risk as an engineering constraint, not a policy debate.

## Concept
Dual-use uplift: the measurable delta between "what a domain expert can do without AI" and "what they can do with AI," across four categories — cyber offensive operations, biological agent design, chemical synthesis planning, and nuclear material processing. Define the threat model, then examine the asymmetry: defensive applications often require less uplift than offensive ones.

## Mechanism
Walk through published capability evaluations (CYBEREVAL, biological knowledge assessments, chemistry tool-use benchmarks). Explain the evaluation methodology: baseline human performance → tool-augmented human → LLM-assisted human. Map where the uplift curve is steep (novel exploit chaining, protocol-aware reconnaissance) versus flat (physical lab execution, fissile material acquisition). Present the four-category risk taxonomy and the mitigations that actually reduce capability (output filtering, refusal training, usage caps) versus security theater.

## Code
Build a dual-use classifier that flags prompt–response pairs against a keyword-and-semantic risk taxonomy. Uses a local embedding similarity check against a small seed set of high-risk phrase clusters across all four domains, with adjustable threshold. Prints per-domain risk scores and a final pass/fail decision.

## Use It
Translate the four-category risk model into a GTM compliance review gate. Map to the AI governance / responsible AI cluster in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`: every AI-assisted workflow that touches prospect data, security tooling, or vertical-specific claims (healthcare, defense, energy) runs through a lightweight dual-use risk checklist before shipping. **Exercise hook (easy):** Write the four-question checklist for a GTM workflow that uses LLM-generated security recommendations. **Exercise hook (medium):** Annotate three real prompt templates from your GTM stack with the risk categories they could trigger. **Exercise hook (hard):** Build a pre-send filter in your automation pipeline that flags outbound AI-generated content crossing a configurable risk threshold, with logging.

## Ship It
Deliverable: a `dual_use_risk_policy.md` document and a working classifier script that integrate into your CI/CD or content review pipeline. The policy defines which of the four categories your organization's AI use cases touch, what uplift scenarios are plausible, and what mitigations are enforced. The classifier runs as a gate on AI-generated outputs. Foundational for Zone 4 (AI Governance & Compliance) — dual-use risk assessment is a prerequisite for responsible GTM operations in regulated verticals.