# Lesson Title: Hierarchical Architecture and Its Failure Mode

## Hook

Why hierarchical decomposition looks like the obvious answer to complex GTM workflows—and why it collapses under real production load. A single supervisor agent routing to three sub-agents sounds clean. In practice, latency compounds, context dilutes at each layer, and one malformed output at depth-3 poisons everything upstream.

## Concept

The mechanism: a tree of LLM calls where each parent synthesizes child outputs into a progressively narrower answer. Traversal is depth-first or breadth-first; each edge adds one full inference round-trip. The failure mode is *contextual dilution*—every synthesis step discards signal, and errors at leaf nodes propagate with no correction path. Contrast with flat fan-out (parallel calls, single merge) and iterative refinement (same call, looped with critique).

## Demo

Build a 3-level hierarchy that researches a company: root dispatches to industry analyst → analyst dispatches to financial reader and news scraper → leaves return raw text → each level summarizes upward. Inject a deliberate hallucination at a leaf node and print the propagated distortion at the root. Observable output: token-by-token trace showing where the hallucination entered and how each synthesis layer amplified or buried it.

## Use It

Map hierarchical decomposition to the Clay enrichment waterfall: sequential data provider calls where each step conditions on the prior result. The failure mode maps directly—if ZoomInfo returns a stale title, the subsequent email waterfall operates on wrong context. Flat fan-out (parallel provider calls, merge after) is the correction pattern. This connects to the **Data Enrichment & Scoring** cluster in the GTM topic map.

## Ship It

Implement a configurable orchestrator that accepts a YAML tree definition, runs the hierarchy, logs per-node latency and token usage, and detects context dilution by comparing leaf outputs to root summary via embedding cosine distance. Three difficulty tiers: (Easy) fixed 2-level tree with hardcoded prompts; (Medium) YAML-driven N-level tree with error logging; (Hard) add automatic collapse detection—when cosine distance between leaf and root drops below threshold, flatten the subtree and re-run.

## Evaluate

Three prompts: (1) Trace a specific failure in a provided execution log and identify which edge introduced the distortion. (2) Given a GTM enrichment sequence, predict where context dilution will cause the most revenue impact. (3) Rewrite a 4-level hierarchy as a flat fan-out with equivalent coverage—justify the trade-off.

---

## Learning Objectives (3–5, action verbs only)

1. **Build** a multi-level LLM call hierarchy and trace output propagation through each synthesis layer.
2. **Detect** context dilution by comparing leaf-node outputs to root summaries using embedding similarity.
3. **Compare** hierarchical, flat fan-out, and iterative refinement architectures on latency, cost, and error propagation.
4. **Diagnose** a specific failure in a hierarchical GTM enrichment pipeline and propose a structural fix.
5. **Implement** automatic collapse detection that flattens underperforming subtrees at runtime.

## GTM Redirect Rules

- **Use It section**: Maps to the **Data Enrichment & Scoring** cluster—specifically the Clay waterfall pattern where sequential provider calls create the same failure mode as hierarchical LLM orchestration.
- **Ship It section**: The configurable orchestrator is a simplified model of how multi-step enrichment workflows fail in production CRM/revops stacks.
- If the AI concept does not cleanly map further, the redirect is: "foundational for Zone 01—agent orchestration patterns that underpin automated research and enrichment workflows."

---

*Outline complete. Each beat expands to full prose, working code, and exercise text in the lesson document.*