# Lesson Outline: Fairness Criteria — Group, Individual, Counterfactual

---

## Beat 1: Hook

The three families of fairness criteria don't agree with each other. This lesson shows why the Impossibility Theorem of Fairness (Chouldechova, 2017; Kleinberg et al., 2016) makes criterion selection a engineering decision, not a philosophical one — and what breaks when you pick wrong.

---

## Beat 2: Mechanism

**Group fairness** checks parity of some metric (selection rate, false positive rate, etc.) across protected subgroups. **Individual fairness** (Dwork et al., 2012) requires that similar individuals (by a task-specific Lipschitz metric) receive similar outcomes. **Counterfactual fairness** (Kusner et al., 2017) asks whether the outcome would change if a protected attribute were different, holding all causal ancestors constant. Covers the mathematical definitions, what each assumes about the data-generating process, and why no classifier can satisfy all three simultaneously except in degenerate cases.

---

## Beat 3: Implementation

Python code that loads a synthetic lending dataset, computes group fairness metrics (demographic parity difference, equalized odds difference), computes an individual fairness score via pairwise distance comparison, and constructs a counterfactual test by flipping a protected attribute and measuring prediction change. All metrics print to stdout with clear labels.

**Exercise hooks:**
- Easy: Add a second protected attribute and recompute group fairness metrics.
- Medium: Implement a `fairness_report()` function that accepts a criterion type and returns a pass/fail verdict with threshold.
- Hard: Build a counterfactual data generator that creates paired (factual, counterfactual) rows for every individual in the dataset, then evaluate counterfactual fairness on the full set.

---

## Beat 4: Use It

GTM cluster: **ICP Scoring Models (Zone 1 — Target)**. When building lead scoring or ICP qualification models, group fairness criteria detect whether your model systematically excludes certain company sizes, geographies, or industries. [CITATION NEEDED — concept: fairness auditing in B2B scoring models] The individual fairness criterion applies directly: two companies with similar firmographic profiles should receive similar scores — if they don't, the model has learned a spurious proxy. Exercise: run group fairness metrics on a firmographic scoring dataset and identify which industry segment has the largest selection rate disparity.

---

## Beat 5: Ship It

Implement a fairness gate in a CI/CD pipeline: before a model is promoted, compute the selected criterion and block deployment if disparity exceeds a configured threshold. Code writes a JSON fairness report artifact that downstream systems can read. Exercise: configure the gate for demographic parity difference with threshold 0.1 and show a passing and failing run.

---

## Beat 6: Extend It

**Causal fairness criteria** extend counterfactual fairness with do-calculus — Pearl's ladder of causation applied to fairness. **Intersectional fairness** audits across combinations of protected attributes rather than single dimensions. **Fairness under distribution shift** asks whether a criterion satisfied at training time holds at inference time when the population changes. References: Barocas, Hardt, and Narayanan (*Fairness and Machine Learning*, 2019, draft); Mitchell et al., "Model Cards for Model Reporting" (2019).

---

## Learning Objectives

1. Compute demographic parity difference and equalized odds difference for a binary classifier across subgroups.
2. Evaluate individual fairness by comparing outcome consistency for nearest-neighbor pairs in feature space.
3. Construct a counterfactual fairness test by intervening on a protected attribute and measuring prediction change.
4. Explain why no classifier can simultaneously satisfy calibration, equalized odds, and demographic parity (except in trivial cases), citing the specific impossibility result.
5. Configure a deployment gate that blocks model promotion when a chosen fairness criterion exceeds a specified threshold.