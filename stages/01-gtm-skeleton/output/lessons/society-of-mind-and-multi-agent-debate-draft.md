# Society of Mind and Multi-Agent Debate

## Learning Objectives

1. Implement a multi-agent debate loop where N LLM instances critique and revise each other's outputs
2. Compare single-agent vs. multi-agent responses for factual accuracy and reasoning quality
3. Configure agent personas, critique prompts, and termination conditions for a debate system
4. Detect when multi-agent debate degrades output quality vs. single-agent inference
5. Build a multi-perspective GTM research pipeline using adversarial agent roles

---

## Hook It

Marvin Minsky's *Society of Mind* proposes that intelligence emerges from many simple agents with conflicting goals, not one monolithic reasoner. Multi-agent debate applies this to LLMs: have multiple model instances argue against each other, then aggregate. The hook: a single LLM call often produces plausible but wrong answers. Debate exposes those errors through adversarial critique.

---

## Ground It

**Mechanism:** Assign N agents different roles or random seeds. Each generates an initial response. Agents then see each other's outputs and produce critiques. Each agent revises its answer given the critiques. The loop runs for M rounds or until answers converge. A final aggregation step (majority vote, judge selection, or fusion) produces output.

Key concepts to cover:
- Agent persona assignment (supporter, critic, synthesizer)
- Critique prompt engineering — what specifically to attack (facts, logic, assumptions)
- Convergence detection — when are agents agreeing enough to stop?
- Degradation risk — agents can converge on wrong answers through social dynamics, not truth-seeking
- Cost multiplier — N agents × M rounds = N×M API calls minimum

Reference: Du et al. (2023) "Improving Factuality and Reasoning in Language Models through Multiagent Debate" [CITATION NEEDED — concept: multi-agent debate original paper and replication results]

---

## Build It

**Exercise Easy:** Two-agent debate on a factual question. Print initial answers, critiques, and final answers side-by-side. Observe whether answers converge.

**Exercise Medium:** Three-agent debate with assigned personas (optimist, pessimist, neutral analyst). Run 3 rounds. Implement convergence detection (compare final answers for semantic similarity). Print round-by-round answers and convergence metric.

**Exercise Hard:** Full debate system with configurable N agents, M rounds, persona injection, and a judge agent that selects the best final answer with reasoning. Include cost tracking (total tokens across all rounds). Output structured JSON with debate transcript and final verdict.

All code runs in terminal, prints observable output, uses Claude API or a mock that simulates multi-turn debate locally.

---

## Use It

**GTM Cluster:** Zone 1 – Enrichment, specifically multi-perspective ICP research and competitive intelligence.

**Mechanism → GTM redirect:** A multi-agent debate system where one agent argues "this company fits our ICP" and another argues "this company does NOT fit our ICP" produces a more defensible qualification decision than a single classification call. The debate transcript surfaces the specific signals that matter.

**Application:** Run competitive positioning debates. Agent A argues for your product's superiority on a dimension; Agent B argues for the competitor's. The transcript becomes raw material for objection handling docs and battle cards. This is not a generic "AI helps GTM" claim — it is a specific application of adversarial critique to positioning work.

**GTM redirect:** Foundational for Zone 1 enrichment workflows where multiple perspectives on a single company or contact reduce false positives in ICP matching.

---

## Ship It

Production considerations that practitioners must address:
- **Cost control:** 4 agents × 3 rounds = 12 API calls per input. Set budgets. Track tokens.
- **Latency:** Serial debate rounds are slow. Parallel initial generation, serial critique rounds is a common pattern.
- **Degradation detection:** Log whether debate improved or degraded vs. single-agent baseline. Not every input benefits from debate.
- **When NOT to debate:** Simple extraction, formatting tasks, and deterministic lookups. Debate helps for judgment-heavy tasks, not for retrieval-heavy tasks.
- **Aggregation strategy:** Majority vote works for classification. Judge synthesis works for generation. Pick based on output type.

**Exercise hook:** Instrument a debate system with cost/latency metrics and quality scoring. Run 10 test cases through both single-agent and debate. Report which cases debate helped, which it hurt, and the cost delta.

---

## Stretch It

**Architectures beyond flat debate:**
- **Star topology:** One central agent receives critiques from all others, revises centrally
- **Sequential (cascade):** Agent 1 generates, Agent 2 critiques/revises, Agent 3 critiques/revises — no parallel critique
- **Mixture of models:** Use different underlying models per agent (Claude vs. GPT vs. Gemini) to diversify reasoning styles

**Open questions to flag honestly:**
- Does multi-agent debate actually improve accuracy, or does it just increase confidence in the answer? Research is mixed. [CITATION NEEDED — concept: empirical debate quality benchmarks beyond original paper]
- Convergence may indicate groupthink, not correctness. Detection of false convergence is an unsolved problem.

**Exercise hook:** Implement a star-topology debate where a "lead researcher" agent revises its ICP analysis based on critiques from "data skeptic," "market realist," and "champion" agents. Compare output quality to flat multi-agent debate on the same inputs.