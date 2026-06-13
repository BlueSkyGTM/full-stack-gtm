# Scaling Laws

## Hook

You're evaluating whether to fine-tune a 7B model on 50K examples or pay for GPT-4 inference on 500K calls. Scaling laws let you predict which path hits your performance target before you spend a dollar. This is the difference between "try it and see" and "compute the answer."

## Concept

The empirical power-law relationships between training compute, parameter count, dataset size, and downstream loss. Kaplan et al. (2020) established that loss scales predictably as a power law with each factor. Hoffmann et al. (2022) — the Chinchilla paper — corrected the optimal allocation: most organizations were undertraining their largest models. The mechanism is a set of equations: L(N) ≈ (Nc/N)^αN for parameters, L(D) ≈ (Dc/D)^αD for data, with exponents empirically measured near 0.05–0.09. These are not theoretical proofs — they are curve fits to experimental data, and they break down at extreme scales.

## Use It

**GTM Redirect:** Foundational for Zone 1 (Prospecting AI). Scaling laws determine which model tier you select for lead scoring, enrichment, and personalization tasks. If your task needs a Chinchilla-optimal 70B model but you're running 7B, no prompt engineering closes that gap — the loss floor is physically higher. This also budgets your inference spend: you can compute whether that extra 2% accuracy justifies the 10x cost increase from one model tier to the next.

## Build It

**Exercise (Medium):** Given published loss curves for the LLaMA model family (7B, 13B, 30B, 65B), fit a power-law regression in Python. Predict the loss for a hypothetical 100B model trained on the same data budget. Print the predicted loss and compare to the actual reported value for LLaMA-65B to validate the fit. Observable output: the fitted exponents, the predicted 100B loss, and the residual error.

## Ship It

**Exercise (Hard):** Write a cost-performance calculator. Inputs: your API budget ($), cost per 1K tokens for two model tiers (e.g., GPT-4o-mini vs. GPT-4o), and a scaling-law-derived performance estimate for each tier. Output: the break-even volume where the higher-cost model's predicted accuracy gain justifies the spend. Print a recommendation with numbers. This is what you run before approving any AI GTM tooling budget.

## Extend It

- **Chinchilla implications for fine-tuning:** If you're fine-tuning with 10K examples on a 7B model, you're operating far from the compute-optimal frontier. The model is "overparameterized" for your data — which is actually fine for fine-tuning, but you should know why.
- **Inference-time scaling:** [CITATION NEEDED — concept: inference compute scaling laws, o1-style reasoning] — the emerging claim that performance scales with inference-time compute, not just training compute. The mechanism is less well-characterized than training scaling laws.
- **Emergent capabilities debate:** Whether certain capabilities "emerge" sharply at scale or improve smoothly. The current evidence suggests most capabilities improve smoothly when measured with proper metrics; apparent emergence often reflects poor metric choice (Wei et al., 2022; Schaeffer et al., 2023).