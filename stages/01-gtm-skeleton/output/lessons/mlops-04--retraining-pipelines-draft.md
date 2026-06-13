# MLOps 04 — Retraining Pipelines

## Beat 1: Hook

Models decay. The data distribution that produced your training set never stays still. A retraining pipeline is the mechanism that detects when decay matters and automates the response. Without it, you're running a time-limited model and hoping the clock hasn't expired.

---

## Beat 2: Concept

**Drift taxonomy.** Three failure modes: covariate drift (input distribution shifts), concept drift (the relationship between input and output changes), and prediction drift (model output distribution changes even if inputs haven't). Each requires a different detection strategy.

**Retraining triggers.** Three common architectures: scheduled (calendar-based, blind to actual drift), metric-triggered (fires when performance on a labeled window degrades below a threshold), and drift-triggered (fires when a statistical test on inputs or outputs exceeds a threshold). Production systems often combine all three.

**The pipeline shape.** Fetch new data → merge with historical → preprocess → train challenger → validate against champion → promote or discard. The validation gate is the critical piece: a retrained model that hasn't proven superiority over the incumbent should never reach production.

**Champion-challenger protocol.** The current production model is the champion. Any retrained candidate is a challenger. The challenger must win on a held-out evaluation set (or a shadow traffic window) before promotion. This prevents performance regressions from automated retraining.

**Data versioning requirement.** Retraining is irreproducible without versioned datasets and versioned training configs. Every retraining run must be traceable to a specific data snapshot and a specific set of hyperparameters. [CITATION NEEDED — concept: data versioning tools for retraining traceability in MLOps]

---

## Beat 3: Demonstration

Three working code blocks, each producing observable output:

1. **Drift detection using statistical tests.** Apply the Kolmogorov-Smirnov test to two feature distributions (simulated reference vs. current window). Print test statistic, p-value, and drift/no-drift decision.

2. **Automated retraining with a validation gate.** Train a champion model on a base dataset. Simulate new data arrival. Train a challenger model on the combined dataset. Compare F1 scores. Print promotion decision and the metric delta.

3. **Retraining trigger logic.** Implement a function that accepts a performance metric history and a threshold, returns whether retraining should fire. Feed it a degrading metric series. Print the trigger decision at each window.

---

## Beat 4: Use It

**GTM redirect.** This is the mechanism behind model maintenance in Zone C of the GTM topic map — specifically, lead scoring models, intent classification models, and ICP fit models that degrade as market segments shift. A GTM team running a scoring model on inbound leads has an implicit retraining problem: if the ideal customer profile shifts (new vertical, new geography, new pricing tier), the model's training data no longer represents reality. The drift detection and champion-challenger patterns from this lesson are the same patterns that keep GTM models accurate.

**Specific mechanism.** When a Clay waterfall enriches lead data that feeds a scoring model, covariate drift is the risk — the enrichment fields change schema or distribution. When an ICP definition changes, that's concept drift. The retraining pipeline structure (detect → train → validate → promote) applies directly.

---

## Beat 5: Ship It

**Easy.** Run the drift detection code on two provided CSVs (reference and current). Report which features show drift at p < 0.05.

**Medium.** Extend the champion-challenger code to log every retraining decision to a JSON file with timestamp, metric values, and promotion outcome. Run three simulated retraining cycles.

**Hard.** Build a complete retraining pipeline script that: reads a configurable drift threshold from a YAML file, detects drift on a provided dataset, trains a challenger, compares against a saved champion model, and writes a promotion decision to a log. Test it with two scenarios (drift detected, no drift).

---

## Beat 6: Review

**Key mechanisms to retain:**
- Three drift types and their detection strategies
- Champion-challenger validation gate
- The retraining pipeline shape (fetch → merge → train → validate → promote or discard)
- Why scheduled retraining without drift detection is insufficient

**Quiz hooks.** Questions should test: distinguishing drift types given a scenario, reading drift test output to determine retraining trigger, predicting champion-challenger outcome given metric values, identifying the failure mode when validation gates are removed.

---

## Learning Objectives

1. Implement drift detection using statistical tests on feature distributions.
2. Build a retraining pipeline with a champion-challenger validation gate.
3. Configure retraining triggers based on performance degradation thresholds.
4. Compare scheduled, metric-triggered, and drift-triggered retraining strategies.
5. Evaluate whether a retrained model should be promoted to production.