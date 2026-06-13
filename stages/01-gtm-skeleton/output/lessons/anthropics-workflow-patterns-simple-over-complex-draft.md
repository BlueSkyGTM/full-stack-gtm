# Anthropic's Workflow Patterns: Simple Over Complex

---

## Hook

Start with the failure mode: a 12-node agent graph that flakes on edge cases, compared to a two-step chain that ships. Present Anthropic's own guidance — start with the simplest pattern that could work, add structure only when measurably necessary.

---

## Concept

Define the five workflow patterns from Anthropic's "Building Effective Agents" guidance: prompt chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer. Map each to its complexity cost and the specific failure mode it introduces. State the thesis: complexity is a cost, not a feature.

---

## Mechanism

Walk each pattern in ascending complexity. For each: the structural diagram (which steps feed which), the coupling it introduces, the failure mode unique to that pattern, and the diagnostic signal that tells you to graduate to it. Implement each pattern as a callable function that prints its decision trace, so the practitioner can see the execution path. Code examples run unmodified in terminal.

Exercise hooks:
- **Easy**: Implement a prompt chain (two sequential calls) that enriches a company name → industry → ICP score. Print intermediate outputs.
- **Medium**: Build a router that classifies inbound leads into three lanes, then runs lane-specific logic. Print the routing decision and downstream output.
- **Hard**: Implement an evaluator-optimizer loop that critiques its own enrichment output and retries until a quality threshold is met. Print iteration count and final output.

---

## Use It

GTM cluster redirect: **Zone 3 — Enrichment**. The Clay waterfall is a prompt chain — each enrichment step feeds the next, and the chain stops when confidence is sufficient. Routing maps directly to lead/lane routing in outbound workflows (SDR split by segment, vertical, or intent signal). Parallelization maps to multi-source enrichment calls. The practitioner should recognize that most GTM enrichment workflows are chains or routers — not orchestrator-workers — and that adding agentic complexity to a waterfall rarely improves conversion. [CITATION NEEDED — concept: Anthropic workflow patterns applied to enrichment waterfall design]

---

## Ship It

Build a working enrichment pipeline that starts as a single prompt, then graduates to a chain only when the practitioner can demonstrate (with printed output) that the single prompt fails a measurable threshold. The exercise forces the practitioner to justify each complexity addition. Output is a runnable script and a one-sentence justification per pattern upgrade.

Exercise hooks:
- **Medium**: Implement a company enrichment function. Start with one prompt. Add a second step only if the first misses a required field. Print the before/after.
- **Hard**: Take the routing exercise from Mechanism. Add an evaluator step that flags low-confidence routes. Print the evaluator's critique and the corrected routing decision.

---

## Evaluate

Three diagnostic questions the practitioner asks before adding a node to any workflow: (1) What failure does this node prevent? (2) Can I measure that failure? (3) Is the failure rate high enough to justify the coupling cost? If the answer to any is "no," the node doesn't ship. Frame this as the default stance for GTM workflow design — simple until proven complex.