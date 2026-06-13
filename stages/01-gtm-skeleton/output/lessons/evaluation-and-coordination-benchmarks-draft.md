# Evaluation and Coordination Benchmarks

## Beat 1: Hook

You shipped a reply classifier. It works on your five test emails. Then production hits, and "interested" gets tagged on out-of-office replies. Without evaluation benchmarks, you're flying blind on every model swap and prompt change.

## Beat 2: Concept

Define what an evaluation benchmark actually is: a fixed dataset with known labels, a scorer that compares predictions against those labels, and an aggregation method that reduces scores to a single decision metric. Distinguish between static benchmarks (MMLU, HumanEval) and operational benchmarks (your own golden dataset of classified replies). Explain why coordination across multiple evaluation dimensions matters—accuracy on its own hides failure modes that precision-at-k or per-class F1 would reveal.

## Beat 3: Mechanism

Walk through the eval pipeline: golden dataset → inference → scorer → metric → decision threshold. Show how to construct a scorer function that takes (prediction, ground_truth) and returns a numeric signal. Explain coordination as the practice of running multiple scorers in parallel and requiring agreement before promoting a model or prompt change. Reference how structured evaluation frameworks implement this pattern. [CITATION NEEDED — concept: coordination protocols for multi-metric evaluation in production LLM systems]

## Beat 4: Use It

GTM redirect: Cluster 11 — Evaluations, LLM testing → Living GTM. Evaluations are A/B testing your sequences before they go live; reply classification is your eval feedback loop. Build a golden dataset of 50 real sales replies with ground-truth labels (interested, not interested, out-of-office, referral). Run your classifier against it. Track precision per class to catch the specific failure mode that matters for pipeline handoffs. This is the mechanism Gong and similar tools use to justify model updates internally.

## Beat 5: Ship It

Implement a minimal evaluation harness in Python: load a JSONL golden dataset, run inference on each record, compute accuracy and per-class precision/recall, print a summary table. No frameworks, no abstractions—just the raw loop so the mechanism is visible. Then add a coordination gate: if any class F1 drops below 0.7, the eval fails and returns exit code 1. Wire this into a CI check.

## Beat 6: Review

Recap the three components of any eval benchmark (dataset, scorer, aggregation) and the coordination principle (never trust a single metric). Preview next lesson's topic: continuous evaluation in production and drift detection. Leave the practitioner with the rule: no model or prompt ships without passing the benchmark gate.

---

### Exercise Hooks

- **Easy**: Build a golden dataset of 20 items and compute raw accuracy against a toy classifier.
- **Medium**: Add per-class precision/recall to the eval harness and identify which class fails first.
- **Hard**: Implement a coordination gate that compares two prompt variants across three metrics and auto-selects the winner, printing the justification.