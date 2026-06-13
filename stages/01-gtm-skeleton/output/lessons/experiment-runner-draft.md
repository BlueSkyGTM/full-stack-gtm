# Experiment Runner

## Learning Objectives

1. Build an experiment runner that executes multiple variants and collects structured results
2. Implement random assignment and sample splitting for fair variant comparison
3. Detect when experiment results reach statistical significance using a permutation test
4. Configure sample size, confidence thresholds, and stopping conditions
5. Evaluate experiment output to select a winning variant with defensible evidence

---

## Beat 1: Hook

You changed a prompt template in your enrichment workflow. Reply rates went up. Was it the prompt, or was it Tuesday? Without an experiment runner, you're guessing. This lesson builds the machinery that turns "I think this works" into "this works, and here's the evidence."

---

## Beat 2: Concept

**Mechanism: Controlled experiment execution.** An experiment runner takes N variants, assigns each unit (prospect, account, email) to exactly one variant at random, records the outcome, and evaluates whether observed differences are real or noise. Three components: assignment (who sees what), collection (what happened), evaluation (does it matter). The evaluation mechanism is a permutation test — shuffle the labels, recalculate the difference, repeat. If the observed difference rarely appears in shuffled data, it's likely real.

**Key terms:** variant, assignment unit, outcome metric, null distribution, p-value, minimum detectable effect.

---

## Beat 3: Demo

Working Python script that:
- Defines three email subject line variants
- Assigns 300 simulated prospects randomly (100 per variant)
- Simulates open rates with different underlying probabilities
- Runs a permutation test (1000 shuffles) to test if the best variant is statistically different from the second-best
- Prints: raw rates, observed difference, p-value, and a decision ("ship variant B" or "need more data")

All output printed to terminal. No external APIs. Uses only `random` and `collections` from stdlib.

---

## Beat 4: Use It

**GTM Redirect:** This is the mechanism behind A/B testing outbound sequences — the same pattern used to test subject lines, CTA wording, send times, and prompt variants in enrichment workflows. [CITATION NEEDED — concept: GTM A/B testing cluster mapping to outbound sequence optimization]

Walk through mapping the generic runner to a real GTM scenario: you have two Clay enrichment prompts (variant A: extract pain points from 10-K; variant B: extract pain points from earnings call transcripts). The experiment runner assigns accounts to each variant, records whether the SDR found the enrichment useful (binary outcome), and tells you which prompt to ship.

**Exercise hook (Easy):** Modify the demo script to test four variants instead of three. Print a ranked results table.

---

## Beat 5: Ship It

**GTM Redirect:** Ship a reusable experiment runner module that you can import into any GTM workflow — outbound sequences in Salesloft, enrichment prompts in Clay, or ICP scoring weights in your scoring model.

The exercise has the practitioner wrap the demo logic into a `ExperimentRunner` class with methods: `add_variant()`, `assign(unit_id)`, `record(unit_id, outcome)`, `evaluate()`. The class persists results to a JSON file so experiments survive across runs. The practitioner then writes a second script that imports the class, loads a running experiment from disk, adds 50 new observations, and re-evaluates.

**Exercise hook (Hard):** Add a `stopping_rule()` method that returns `"stop"` when the best variant is significant at p < 0.05 with a minimum sample of 200 per variant, and `"continue"` otherwise. Demonstrate early stopping on a simulated experiment.

---

## Beat 6: Evaluate

Three questions, grounded in the mechanism:

1. Your permutation test returns p = 0.03 on variant A vs B. Variant A has 12% conversion, variant B has 11.5% conversion, and each has 80 observations. Should you ship variant A? Explain why or why not. *(Tests: sample size awareness, practical vs statistical significance)*

2. You add a third variant mid-experiment. The first two variants each have 200 observations; the third has 40. Your runner reports the third variant is winning with p = 0.01. What went wrong? *(Tests: assignment integrity, unequal sample sizes, peeking)*

3. Write the three lines of code that randomly assign a unit to one of K variants with equal probability, without using any external libraries. *(Tests: random assignment mechanism)*

---

**GTM Cluster Note:** The experiment runner pattern is foundational for any GTM team running systematic tests on outreach, enrichment, or scoring. If a specific GTM topic map cluster for "A/B testing" or "experimentation" exists in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`, the "Use It" and "Ship It" sections should redirect to that cluster explicitly. The current citation placeholder reflects that this mapping needs confirmation.