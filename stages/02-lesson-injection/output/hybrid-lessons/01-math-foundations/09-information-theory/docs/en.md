# Information Theory

## Learning Objectives

- Compute entropy, cross-entropy, and KL divergence from raw probability distributions using Python, without relying on framework abstractions.
- Derive the equivalence between minimizing cross-entropy loss and maximizing log-likelihood, and trace this through a concrete numerical example.
- Implement mutual information between a categorical feature and a target label, then use it to rank features by predictive signal.
- Evaluate data fields by information content and identify degenerate (near-zero-entropy) columns that waste enrichment budget.
- Build a CLI tool that audits a CSV for column entropy and mutual information with a target, producing a ranked report.

## The Problem

Every time you call `CrossEntropyLoss()` in PyTorch, you are running a 1948 formula invented by Claude Shannon to measure telephone line noise. Every time an LLM outputs a token, it is selecting the token that minimizes cross-entropy between its predicted distribution and the training data's distribution. Every time you read "perplexity" in a model card, you are looking at exponentiated entropy. These are not separate ideas — they are one idea wearing different uniforms.

The practical problem is this: if you do not understand what cross-entropy actually measures, you cannot diagnose why your model's loss plateaus, why your classifier assigns everything to one class, or why your prompt produces generic output instead of specific output. You are calling a function whose behavior you cannot inspect.

Information theory gives you the vocabulary. Surprise is the negative log of a probability. Entropy is the average surprise of a distribution. Cross-entropy is the surprise you experience when reality follows one distribution but your model assumed another. KL divergence is the gap between those two — the extra penalty for being wrong. Mutual information is how much knowing one variable reduces your uncertainty about another. Every loss function, every evaluation metric, and every feature-selection method in machine learning is built from these four quantities.

## The Concept

### Information Content (Surprisal)

When something unlikely happens, it carries more information. A coin landing heads is expected — one bit of surprise at most. A 1-in-1000 event actually happening is far more surprising. The information content of an event with probability $p$ is:

$$I(x) = -\log_2 p(x) \text{ (bits)}$$

or equivalently $-\ln p(x)$ in nats. The base changes the unit, not the relationship.

| Event | Probability | Surprise (