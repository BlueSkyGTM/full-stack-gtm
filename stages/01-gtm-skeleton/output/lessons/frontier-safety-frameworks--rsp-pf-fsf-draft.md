# Frontier Safety Frameworks — RSP, PF, FSF

## Hook
Anthropic, OpenAI, and Google DeepMind each published internal governance frameworks in 2023–2024 that define when a frontier model is too dangerous to ship. If you build on these models, these frameworks determine what capabilities get deployed, what gets delayed, and what gets blocked entirely.

## Concept
Each framework shares a common skeleton: define capability thresholds → evaluate models → apply mitigations → gate deployment. RSP (Anthropic) anchors to ASL levels adapted from biosafety. PF (OpenAI) anchors to risk scores across threat categories. FSF (Google DeepMind) anchors to critical capability evaluations with layered mitigations. The differences matter because they produce different deployment decisions for the same model capability. [CITATION NEEDED — concept: comparative analysis of RSP vs PF vs FSF threshold definitions]

## Mechanism
Walk through the threshold-evaluation-mitigation-gate pipeline shared by all three frameworks. Then map where they diverge: RSP uses evals as hard gates with voluntary commitments, PF uses risk scorecards with continuous monitoring, FSF uses capability evaluations with both model-level and system-level mitigations. Show what each framework would do with a hypothetical model that demonstrates strong persuasion capability.

## Lab
**Medium exercise:** Given a mock evaluation result set (CBRN knowledge, cyber offensive capability, persuasion, autonomy), classify the model under each framework's threshold system and determine whether deployment would be allowed, allowed with mitigations, or blocked. Print the classification and reasoning for each framework.

## Use It
These frameworks govern what frontier capabilities are available to GTM tooling built on Claude, GPT, and Gemini. When designing AI-powered outreach, research, or personalization workflows, the safety evaluations behind these frameworks determine what the model will refuse, what it will gate behind safety training, and what it will serve directly. This is foundational for **Zone 02 (Enrichment & Research)** and **Zone 04 (Outreach & Engagement)** — the persuasion and autonomy evals directly constrain how aggressively an AI SDR can operate. [CITATION NEEDED — concept: how frontier safety evals constrain GTM AI tool behavior in production]

## Ship It
Build a one-page safety-check reference for your team's AI stack: for each model provider you use, document which framework governs it, what capability thresholds are most relevant to your use case (persuasion for outreach, autonomy for agents), and what mitigations are already applied at the API layer. This becomes a living document you update when providers publish framework revisions.