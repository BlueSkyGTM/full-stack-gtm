# Model, System, and Dataset Cards

## Hook

A model ships to production. Three months later, nobody remembers what training data it used, what "fairness" meant to the team, or why the threshold was set at 0.73. Cards solve the institutional memory problem by making documentation a structured, version-controlled artifact rather than an afterthought.

## Concept

**Beat description:** Three card types, three scopes. Model cards document a single trained artifact: weights, benchmarks, intended use, failure modes. Dataset cards document the provenance, composition, and licensing of training or evaluation data. System cards document an assembled pipeline: which models call which, what guardrails wrap them, how failure propagates. The mechanism is constraint: by forcing answers to fixed questions (intended use, out-of-scope use, metrics, ethical review), cards prevent the documentation drift that makes deployed systems unaccountable. [CITATION NEEDED — concept: Model Cards for Model Reporting, Mitchell et al. 2019; Data Sheets for Datasets, Gebru et al. 2021]

## Demo

**Beat description:** Generate a minimal but valid model card as YAML, validate it against a schema, and print the parsed fields. Demonstrates that cards are structured data, not prose — they can be linted, diffed, and programmatically consumed.

## Use It

**Beat description:** Write a dataset card for a prospect scoring dataset (the kind used in enrichment pipelines), including data provenance, license constraints, and known biases. The GTM redirect: this is the documentation layer behind any scoring model in the Clay waterfall — without it, you cannot explain why an ICP score changed after a data source swap.

**Exercise hooks:**
- Easy: Fill in a pre-structured dataset card template for a provided CSV of firmographic data.
- Medium: Write a model card for a logistic regression lead-score model, including the evaluation metric and threshold justification.
- Hard: Write a system card for a multi-step enrichment → score → personalize pipeline. Document failure propagation: what happens when the enrichment step returns partial data.

## Ship It

**Beat description:** Check a model card into a repo alongside the trained model artifact, wire a CI check that fails if the card's `last_reviewed` date is older than 90 days, and confirm the check runs. Cards that live outside the deployment pipeline are decoration; cards that block a deploy are infrastructure.

**Exercise hooks:**
- Easy: Add a YAML model card file to a repo and confirm it parses without error.
- Medium: Write a script that reads all model card YAMLs in a directory and prints which ones have stale review dates.
- Hard: Write a GitHub Action that fails a PR if any model card is missing required fields or has a `last_reviewed` date older than a configurable threshold.

## Reference

**Beat description:** Original paper citations for Model Cards (Mitchell et al., 2019), Data Sheets for Datasets (Gebru et al., 2021), and System Cards (industry practice — no single canonical paper; cite Hugging Face's implementation as a reference artifact). Link to the Hugging Face model card template and the Dataset Card specification used in their Hub.

---

**Learning Objectives (draft):**

1. Write a valid model card that documents intended use, out-of-scope use, evaluation metrics, and known limitations for a trained model.
2. Write a dataset card that captures provenance, composition, licensing, and collection methodology for a training or evaluation dataset.
3. Write a system card that documents component interaction, failure propagation, and guardrails for a multi-step AI pipeline.
4. Implement a CI check that validates card completeness and staleness before a model artifact can deploy.
5. Explain the institutional memory problem that cards solve and the risks of unstructured or absent documentation in deployed systems.