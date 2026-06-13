# Scalable Oversight and Weak-to-Strong Generalization

## Hook: Why It Matters

When the system you're supervising can out-think you in the domain you're evaluating, your oversight signal degrades. This is the core problem behind scalable oversight — and by extension, any production AI system that exceeds the checker's capability. Weak-to-strong generalization asks the empirical question: can a weaker model extract useful supervision signal over a stronger one?

---

## Beat 1: Connect

Build intuition for why supervision breaks at capability asymmetry. Present the "sandwich" problem: human evaluates AI, AI evaluates AI, both may fail on the same class of inputs. Frame why this matters for any practitioner running multi-model pipelines where output quality is critical.

---

## Beat 2: Explain

**Scalable Oversight** — family of techniques (debate, recursive reward modeling, iterated amplification) designed to extend human oversight beyond human evaluation capacity. Mechanism: decompose hard evaluation into easier sub-evaluations, or pit models against each other to surface failures.

**Weak-to-Strong Generalization** — specific research program (OpenAI Superalignment, Burns et al. 2023) testing whether a weaker supervisor model can provide training signal to a stronger student. Mechanism: train a weak model on ground truth, use its predictions as labels for a stronger model, measure whether the strong model recovers super-weak performance or collapses toward the weak ceiling. Key finding: strong models can generalize beyond weak supervision on some tasks, but not all — the gap is task-dependent and not yet predictable.

[CITATION NEEDED — concept: quantitative results on weak-to-strong generalization gaps across task families]

---

## Beat 3: Demonstrate

Build a minimal weak-to-strong supervision pipeline. Small model generates labels on a held-out set. Large model trains on those labels. Measure: does the large model exceed the weak supervisor's accuracy, or does it merely imitate it? Print metrics that expose the generalization gap.

Exercise hooks:
- **Easy**: Run the provided pipeline, report the generalization gap for one task.
- **Medium**: Swap in a different weak supervisor and compare how gap changes.
- **Hard**: Implement a debate-style oversight variant where two weak models argue before the strong model receives the label.

---

## Beat 4: GTM Redirect — Use It

**GTM Cluster**: Zone 01 — Signal & ICP enrichment; Zone 02 — Quality gates in AI-powered outbound.

In GTM pipelines that generate personalized content, lead scores, or enrichment at volumes no human can review, you are already running a weak-to-strong problem: a cheaper model or rule-based check is supervising a more capable model's output. This lesson's mechanism maps directly to **quality gate design** in automated GTM workflows.

Concrete application: when a strong model writes personalized outreach emails at scale, a weaker/faster model acts as a binary quality filter. The weak-to-strong framework tells you when that filter helps and when it silently collapses — approving everything or rejecting the wrong cases. Build the filter wrong, and your "oversight" is theater.

This is foundational for any Zone 01/02 pipeline where model-generated signal feeds downstream action.

---

## Beat 5: Ship It

Integrate a weak-overseer quality gate into a mock GTM enrichment pipeline. Generate synthetic company profiles with a "strong" model, run them through a "weak" classifier, measure false-accept and false-reject rates against ground truth labels. Output a JSON report of failure modes.

Exercise hooks:
- **Easy**: Ship the pipeline, identify which failure category dominates.
- **Medium**: Tune the weak overseer's threshold to minimize a cost function (false accepts weighted higher than false rejects — simulate "bad email sent" vs "good email not sent").
- **Hard**: Add a second weak overseer in an ensemble and measure whether debate-style disagreement improves the gate.

---

## Beat 6: Extend

Open problems: the generalization gap is not predictable a priori. Current research lacks a theory that says "for task X with capability gap Y, expect Z% supervision degradation." Connections to interpretability (if the strong model's internal representations diverge from the weak supervisor's, the signal breaks). Connections to evals (you can't measure what you can't oversee).

---

## Learning Objectives

1. **Detect** capability asymmetry in a multi-model pipeline and predict where oversight signal will degrade.
2. **Implement** a weak-to-strong supervision experiment and measure the generalization gap.
3. **Compare** scalable oversight methods (debate, recursive reward modeling, amplification) on their assumptions and failure modes.
4. **Configure** a weak-overseer quality gate for a GTM content pipeline and evaluate its false-accept/false-reject tradeoff.
5. **Evaluate** whether a given production system requires scalable oversight techniques or whether standard evaluation suffices.