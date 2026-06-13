# Model Routing as a Cost-Reduction Primitive

## Hook (Why This Matters)

Every LLM call costs money. Most production workloads send every request to the same model regardless of complexity. A query like "summarize this in 3 bullets" hits GPT-4 just like "analyze the competitive dynamics of this market." The first request is overkill at GPT-4 pricing; the second might need it. Model routing is the practice of classifying request complexity before selecting which model handles it. Teams that implement routing report 40-70% cost reductions with negligible quality loss [CITATION NEEDED — concept: production cost reduction benchmarks for model routing]. This lesson builds the classifier that makes that decision.

---

## Concept (Mechanism)

The core mechanism is a **two-stage pipeline**: classify, then invoke. A request arrives. A lightweight classifier (rule, embedding, or tiny model) tags it as `simple`, `medium`, or `complex`. A router function maps that tag to a specific model. The expensive model only runs when the classifier says it should.

Three classification approaches, in order of sophistication:

1. **Rule-based routing.** Keyword matching, regex, or string length heuristics. Zero overhead, brittle edge cases. Works for obvious splits like "translate this" vs. "write a strategic plan."

2. **Embedding-based routing.** Compute cosine similarity between the incoming request and a set of prototype embeddings for each complexity tier. The closest prototype wins. More adaptive than rules, requires maintaining a prototype bank.

3. **Classifier-model routing.** A small, fast model (e.g., Haiku, or a locally-hosted classifier) reads the prompt and outputs a routing label. Most flexible, adds latency and a small per-request cost for the classification call itself.

The key tradeoff: routing accuracy vs. routing overhead. If your classifier costs $0.001 per request and misroutes 10% of queries, the misrouting cost must be lower than the savings from correct routing. That math determines which approach fits your workload.

---

## Demo (Working Code)

Build a working router that classifies prompts by complexity using embedding similarity, routes to mock model endpoints (simulated for zero-cost execution), and prints the routing decision plus estimated cost for each request. The demo accepts a list of test prompts spanning simple (formatting, extraction) to complex (analysis, strategy) and outputs a routing table showing which model each prompt would hit and why.

Observable output: a printed table of prompt → complexity tier → selected model → estimated cost per 1K tokens.

---

## Use It (GTM Redirect)

This lesson's routing mechanism maps directly to **Cluster 06: Embeddings, semantic search** in the Inbound-Led Outbound (3.3) / Signal Machine cluster.

The application: when inbound signals arrive (form fills, email replies, chat transcripts), an embedding-based router classifies intent and routes each lead to the appropriate outreach sequence. A "pricing question" embedding routes to the sales-engineered FAQ sequence. A "competitive comparison" embedding routes to the battle-card sequence. A "just browsing" embedding routes to the nurture drip.

This is model routing applied to GTM workflow triage. Same mechanism — classify, then route to the right handler. The classifier embedding bank becomes your "intent prototype bank," and each outbound sequence is the equivalent of a model endpoint.

**Exercise hook (easy):** Build a prototype bank with 5 intent categories relevant to your GTM motion. Classify 10 sample inbound messages against it. Print the routing decisions.

---

## Ship It (Production Exercise)

Implement a router that logs every routing decision, tracks cumulative cost savings vs. a "send everything to the expensive model" baseline, and handles edge cases (empty prompts, ambiguous classifications, rate limits on the fallback model). The router must expose a single `route(prompt: str) -> dict` interface that returns the selected model, the classification confidence, and the cumulative savings counter.

**Exercise hook (hard):** Instrument the router with a simple JSON log file. Run 50 diverse prompts through it. Generate a summary report showing: total estimated cost with routing vs. without, accuracy spot-check (manually verify 10 routing decisions), and latency overhead from the classification step.

---

## Review (Synthesis)

Model routing converts a fixed-cost inference pipeline into a variable-cost one. The lever is classification accuracy — a bad router is just an expensive router that produces worse outputs. The three approaches (rule, embedding, classifier-model) form a spectrum from zero-cost/zero-adaptability to marginal-cost/high-adaptability. Pick based on your query distribution: if 80% of your traffic is simple, even a rule-based router saves significant money. If your traffic is heterogeneous, embedding-based routing with a maintained prototype bank is the practical middle ground.

The GTM mapping holds because lead triage is the same problem shape: heterogeneous inputs, multiple handlers, cost (human attention) that scales with misrouting. Embedding-based routing transfers directly.

**Discussion prompt:** Where in your current GTM stack is every lead hitting the "most expensive model" (a senior AE) when a cheaper handler (an automated sequence, an SDR, a self-serve flow) would work? That's your routing opportunity.

---

## Learning Objectives

1. **Implement** a two-stage classify-then-invoke routing pipeline that reduces simulated inference cost by at least 30% on a provided test set.
2. **Compare** rule-based, embedding-based, and classifier-model routing on accuracy, latency, and overhead using observable output from working code.
3. **Configure** an embedding prototype bank for a GTM intent-classification task and evaluate routing accuracy against labeled inbound messages.
4. **Calculate** the break-even point where routing overhead exceeds cost savings, given per-request pricing for classification and inference.
5. **Diagnose** misrouting failures from logged routing decisions and propose corrections to the prototype bank or rule set.