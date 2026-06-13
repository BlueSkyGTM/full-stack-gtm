# Information Theory

## Hook It

Every time an LLM generates a token, it is minimizing cross-entropy. Information theory is not abstract math — it is the loss function, the training objective, and the reason your prompts work or fail.

## Name It

Define Shannon entropy, information content (surprisal), cross-entropy, Kullback-Leibler divergence, and mutual information. Each concept gets a one-sentence mechanism definition and a formula.

## See It

Compute entropy for three distributions (uniform, peaked, degenerate) using Python and print the results. Show how cross-entropy penalizes a model that assigns zero probability to a token that actually appears. All output observable in terminal.

**Exercise hook (easy):** Given two probability distributions printed by the script, identify which has higher entropy and explain why in one sentence.

## Try It

Implement cross-entropy loss from scratch on a toy vocabulary of 10 tokens. Compare against PyTorch's `CrossEntropyLoss` to confirm numerical equivalence. Then compute KL divergence between two text corpora represented as token frequency distributions.

**Exercise hook (medium):** Build a function that takes two probability distributions as plain lists and returns their KL divergence. Print the result for three pairs of distributions.

**Exercise hook (hard):** Load a real text file, tokenize it into words, compute per-character entropy, and compare it to a shuffled version of the same text. Print both values and the ratio.

## Use It

Entropy measures how uncertain a distribution is — which maps directly to data enrichment prioritization. Fields with low entropy (same value for most records) carry less information and are lower priority for verification. Mutual information between a firmographic field and conversion outcome tells you which signals are worth buying.

This is foundational for **Zone 1 (Data Foundation)** — specifically, scoring data fields by information gain rather than assumed importance [CITATION NEEDED — concept: entropy-based data enrichment prioritization in GTM workflows].

**Exercise hook (medium):** Given a CSV of 50 company records with 5 firmographic fields, compute the entropy of each field and rank them by information content. Print the ranked list.

## Ship It

Build a CLI tool that reads a CSV, computes entropy per column, flags near-zero-entropy columns (degenerate — almost same value everywhere), and outputs a ranked report. This is a data audit primitive.

**GTM redirect:** This entropy audit is the first pass in any enrichment strategy — you do not buy data for fields that are already deterministic. Foundational for Zone 1 data quality workflows. The same mutual-information computation, applied to conversion labels, becomes feature selection for predictive lead scoring.

**Exercise hook (hard):** Extend the CLI tool to accept a target column name, compute mutual information between every other column and the target, and print a ranked list of predictive signals. Use only standard library plus `numpy`.