# Lesson 40: Direct Preference Optimization from Scratch

## Beat 1: Hook — The Reward Model Is Expensive

 RLHF trains a reward model, then runs PPO against it. Two training loops, four models in memory (policy, reference, reward, critic). DPO collapses this into a single loop by deriving what the optimal policy looks like under the Bradley-Terry preference model. Practitioner starts with the intuition: if you already have preference pairs, why not skip the middleman?

## Beat 2: Concept — Derive the DPO Objective

Walk through the mathematical derivation: start from the KL-constrained reward maximization objective, apply the Bradley-Terry assumption, solve for the implicit reward in terms of the policy and reference policy, substitute back. Result: a loss function that operates directly on `(prompt, chosen, rejected)` triples. No reward model. Key terms: reference log-probs, chosen vs. rejected, the log-sigmoid structure.

## Beat 3: Demonstration — Build the DPO Loss and Train on Synthetic Preference Data

Implement the DPO loss function from scratch in PyTorch. Generate a synthetic binary classification task with preference pairs (correct label = chosen, incorrect = rejected). Train a small transformer or MLP using the DPO loss. Print training loss, accuracy on chosen vs. rejected, and log-probability divergence between policy and reference. Observable output: loss decreasing, chosen log-probs increasing relative to rejected.

## Beat 4: Use It — Preference-Driven Message Ranking in GTM Pipelines

GTM cluster: **Zone 2 — Enrich & Score** and **Zone 3 — Engage**. DPO trains a model to prefer output A over output B given the same input. In GTM, this is message ranking: given a lead profile, prefer the email variant that generated a reply. Build a preference dataset from reply/no-reply signals, then apply DPO to a copy-generation model so it internalizes "what gets replies" without maintaining a separate scoring model. Redirect: this replaces a two-stage "generate then score" pipeline with a single model that natively prefers high-converting outputs.

## Beat 5: Ship It — Preference Data Quality, Reference Model Selection, and Evaluation

Production DPO fails on noisy preferences. Cover: how to filter ambiguous pairs, why the reference model choice matters (it anchors the KL penalty), how to evaluate with win-rate benchmarks. Discuss the temptation to re-label with your own model (label leakage) and how to detect it. Exercise hooks: easy — identify data quality issues in a provided preference dataset; medium — implement a win-rate evaluation loop; hard — build a self-play data generation pipeline that avoids reward hacking.

## Beat 6: Evaluate — Confirm the Mechanism Is Internalized

Assessment targets: (1) explain why DPO eliminates the reward model without changing the theoretical optimum, (2) trace the gradient signal through the DPO loss to confirm it increases chosen log-probs and decreases rejected log-probs relative to reference, (3) diagnose a training run where loss plateaus but win-rate does not improve (typically: preference noise or insufficient KL penalty). Exercise hook: given a failing DPO training log, identify the most likely cause from three candidates and justify.

---

**GTM Redirect:** Zone 2/3 — preference-driven message ranking replaces "generate then score" with a single model that internalizes conversion signal.

**Citations:** [CITATION NEEDED — concept: DPO application to marketing copy optimization in production] for the GTM redirect. The mathematical derivation references Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."