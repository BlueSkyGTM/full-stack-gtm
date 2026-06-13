# Darwin Godel Machine — Open-Ended Self-Modifying Agents

## Hook

The hardest problem in agent design isn't choosing the right model or prompt — it's that the space of possible agent architectures is unbounded, and you can't hand-design what you haven't imagined. The Darwin Godel Machine (DGM) treats the agent's own source code as the search space, letting evolutionary pressure discover architectures no human would write.

## Concept

DGM combines two ideas: open-ended search (Darwin) and self-reference (Gödel). An agent maintains an archive of candidate agent implementations. Each candidate can read and rewrite its own code. Selection pressure comes from performance on a task distribution — not a single benchmark, but an open-ended stream. The agent that modifies itself into a better searcher wins. [CITATION NEEDED — concept: Darwin Godel Machine original paper and architecture]

## Mechanism

Break down the DGM loop: (1) sample a parent agent from the archive weighted by historical performance, (2) the parent proposes a mutated version of its own code, (3) the child is evaluated on a held-out task slice, (4) if the child outperforms, it enters the archive alongside the parent — never replacing it. This is an evolutionary tree, not a hill-climbing path. The key insight: because agents can modify *how they search*, the search strategy itself improves over time. This is meta-learning without a fixed meta-objective. [CITATION NEEDED — concept: DGM mutation operators and archive selection]

## Code

Implement a minimal self-modifying agent loop. A Python function that scores leads, evaluated on a synthetic dataset. The agent can rewrite its own scoring logic via string manipulation of its source. An outer evolutionary loop selects the best rewriter over N generations. Observable output: generation number, fitness score, and the mutated function source at each step.

## Use It

This is foundational for Zone 1 — Intelligence Infrastructure. The specific GTM mechanism: self-optimizing enrichment pipelines. A Clay waterfall that reorders its enrichment steps based on which sequence produced the highest email reply rate is a degenerate single-step DGM. The full pattern — agent rewrites its own routing logic based on conversion signal — maps to ICP refinement loops in outbound. You wouldn't deploy open-ended self-modification in production (undocumented behavior risk), but the *selection archive* pattern applies: keep every version of your scoring model, evaluate each against actual pipeline outcomes, and promote winners without deleting ancestors.

## Ship It

Three exercises at escalating fidelity:

- **Easy**: Implement a two-generation mutation loop on a pure function. Observable: before/after source and score delta.
- **Medium**: Build an archive of 10 candidate agents with fitness-weighted parent sampling. Observable: archive diversity metric after 50 generations.
- **Hard**: Wire the evolutionary loop to a real GTM signal — feed it historical email open rates as the fitness function, with the agent rewriting its own subject-line scoring weights. Observable: generation-over-generation improvement in predicted vs actual open rate correlation. Deploy to staging with a kill switch on regression.

---

**GTM Redirect**: Zone 1 — Intelligence Infrastructure, specifically the ICP refinement and enrichment-optimization loop pattern. The archive-based evolutionary selection maps directly to multi-variant scoring model management in GTM tooling.