# Bias-Variance Tradeoff

## Beat 1: Hook

You built a lead-scoring model. It aces your training data and face-plants on next month's leads. That gap is the bias-variance tradeoff in motion — and tuning it is the difference between a model you trust and a model you delete.

## Beat 2: Concept

Define bias (systematic underfitting, the model is too simple to capture the pattern) and variance (sensitivity to training noise, the model is too complex for the data you have). Show the decomposition: irreducible error + bias² + variance = total error. The tradeoff: reducing one usually inflates the other.

## Beat 3: Mechanism

Walk through three polynomial regression fits on the same dataset — degree 1 (high bias, low variance), degree 15 (low bias, high variance), and the sweet spot (degree 3–5). Compute MSE on train vs. a held-out set. Show learning curves: training error and validation error as functions of data size. Explain why more data shrinks variance but never fixes bias.

## Beat 4: Implementation

Write a working Python script that generates a noisy sinusoidal dataset, fits polynomials of degree 1 through 20, and prints train MSE and test MSE for each degree. The output is a table where you can see test MSE drop, then rise — the U-curve. No matplotlib; pure terminal output.

## Beat 5: Use It

**GTM Redirect:** Zone 1 — ICP Scoring and Enrichment. When you build an ICP fit score using firmographic signals, feature count vs. account count is a bias-variance problem. Too few signals: your score treats every account the same (high bias). Too many signals for 50 closed-won accounts: your score memorizes noise (high variance). Regularization (L1/L2) is the control knob. In Clay, the waterfall enrichment sequence implicitly manages this — each enrichment step adds a feature. If your conversion dataset has N rows and your waterfall pulls M attributes, the ratio N/M dictates whether you should simplify your scoring logic or enrich more accounts before training. [CITATION NEEDED — concept: Clay waterfall feature-to-row ratio heuristics]

Exercise hooks:
- **Easy:** Run the provided script. Identify the optimal polynomial degree from the printed table.
- **Medium:** Modify the script to add L2 regularization (ridge regression) and observe how the U-curve flattens.
- **Hard:** Replace the sinusoidal dataset with a CSV of firmographic features and a binary ICP label. Report which degree overfits fastest and why.

## Beat 6: Ship It

Diagnose your next scoring model before deploying it. If training accuracy is low, the model is underfit — add features or increase complexity. If training accuracy is high but holdout accuracy is low, the model is overfit — regularize, reduce features, or get more data. The heuristic: never trust a model trained on fewer than 10 rows per feature. Write this diagnostic into your scoring pipeline as a gating check.