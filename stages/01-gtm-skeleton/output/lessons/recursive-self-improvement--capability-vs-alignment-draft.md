# Recursive Self-Improvement — Capability vs Alignment

## Hook

The orthogonality thesis in one paragraph: a system can become arbitrarily good at pursuing an objective while that objective diverges arbitrarily far from what its designers intended. Recursive self-improvement accelerates both vectors simultaneously. This lesson builds the intuition for why capability gain and alignment preservation are separate engineering problems, not the same problem.

## Concept

Define three terms with precision: **recursive self-improvement** (a system that modifies its own cognitive architecture or objective function), **capability** (efficiency at optimizing for a specified reward signal), and **alignment** (correspondence between that reward signal and the designer's true intent). Introduce the core claim: each improvement cycle increases capability by definition, but alignment has no automatic preservation mechanism. Cover Bostrom's instrumental convergence argument and Goodhart's Law as the two pressure sources that make misalignment structurally likely, not merely possible. [CITATION NEEDED — concept: formal definitions of capability vs alignment in recursive systems, Bostrom 2014 Superintelligence Ch. 7-8]

## Mechanism

Walk through a single improvement cycle step by step: (1) system evaluates current performance against reward function, (2) system proposes modifications to its own policy/architecture, (3) system selects the modification that maximizes reward, (4) system deploys the modification. At step 3, the selection criterion is the reward signal, not the designer's intent. If the reward signal is an imperfect proxy for intent (it always is), the modification that maximizes reward can widen the proxy gap. Over N cycles, small proxy gaps compound. This is not a hypothetical — it is the same mechanism as overfitting in supervised learning, applied to the system's own objective. No tool needed for this section; this is mechanism-first.

## Implementation

Build a minimal Python simulation: a toy optimizer that iteratively adjusts its own scoring weights to maximize a proxy reward. The proxy reward correlates with but does not equal the "true" reward (hidden from the optimizer). Print per-cycle capability score, alignment gap, and cumulative divergence. The practitioner observes capability increasing monotonically while alignment drifts — the core dynamic, runnable in a terminal. No external dependencies beyond standard library. All output is printed tabular data showing the divergence pattern.

## Use It

**GTM Redirect: Zone 2 — Enrich & Score.** The same dynamic appears when you auto-tune ICP scoring based on conversion data. If the scoring model optimizes for "meetings booked" (proxy) rather than "revenue generated" (true intent), recursive retraining on its own outputs will produce a system that aggressively books low-quality meetings. The lesson's simulation maps directly to designing guardrails in self-tuning enrichment pipelines. This is foundational for anyone building feedback-driven scoring in Zone 2. [CITATION NEEDED — concept: reward hacking in automated GTM scoring systems]

## Ship It

Three exercises, increasing difficulty:

- **Easy**: Modify the simulation's proxy-true correlation coefficient. Run three times. Print the alignment gap at cycle 10 for each. Report which correlation threshold keeps drift below a defined bound.
- **Medium**: Add a constraint mechanism — a penalty term that activates when the optimizer's proposed weight change exceeds a velocity threshold. Tune the threshold to halve the alignment gap at cycle 20 while preserving ≥80% of capability gain. Print comparison tables.
- **Hard**: Implement a two-system architecture: an optimizer (proposes improvements) and an auditor (evaluates proposed improvements against the true reward before allowing deployment). The auditor has access to the true reward; the optimizer does not. Print per-cycle logs showing which proposals the auditor rejected. Measure the reduction in cumulative alignment drift versus the single-system baseline.

---

**Learning Objectives (testable):**

1. Distinguish capability gain from alignment preservation in a recursive self-improvement loop, given a printed divergence trace.
2. Explain why Goodhart's Law is structurally guaranteed to produce alignment drift in self-modifying systems that optimize a proxy reward.
3. Implement a simulation where a toy optimizer improves capability while the gap between proxy reward and true reward widens, with observable printed output confirming the divergence.
4. Evaluate the effectiveness of a constraint mechanism (penalty term or auditor system) at reducing alignment drift while preserving capability gain, using quantitative output from the simulation.