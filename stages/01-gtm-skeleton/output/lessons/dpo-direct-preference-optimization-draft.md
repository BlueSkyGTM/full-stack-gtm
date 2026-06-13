# DPO: Direct Preference Optimization

---

## Beat 1: Hook

**What it is:** RLHF works but requires four moving parts (reference model, reward model, policy model, KL constraint). DPO collapses this to two: the model and preference data. The insight: under the RLHF optimal policy, the reward function has a closed-form solution in terms of the policy itself. So skip the reward model entirely.

**Why now:** Fine-tuning a model to prefer your GTM team's output style over generic responses requires alignment. DPO is the most accessible path—no reward model training, no reinforcement learning loop, just labeled pairs.

---

## Beat 2: Concept

**Mechanism:** DPO reparameterizes the KL-constrained reward maximization problem. Instead of learning a reward function and then optimizing policy against it, DPO solves for the optimal policy directly using a classification loss on preference pairs (chosen vs. rejected).

**Key equation (simplified):** The DPO loss increases log-probability of chosen responses while decreasing log-probability of rejected responses, weighted by how much the reference model already prefers one over the other. The reference model acts as an implicit reward baseline.

**Contrast to RLHF:**
- RLHF: data → reward model → RL optimization → policy
- DPO: data → direct policy update via classification loss

---

## Beat 3: Mechanism

**Step-by-step:**
1. Start with a supervised fine-tuned model (your SFT checkpoint). This becomes both the reference model (frozen) and the policy model (trainable).
2. Format preference data as (prompt, chosen_response, rejected_response) triples.
3. For each triple, compute log-probabilities under both the reference and policy models.
4. Apply the DPO loss: `loss = -log(sigmoid(beta * (log(pi_chosen/pi_rejected) - log(ref_chosen/ref_rejected))))` where `beta` controls the KL penalty strength.
5. Backpropagate. The model shifts toward chosen and away from rejected, moderated by what the reference model already knows.

**The beta parameter:** Low beta = weak preference signal, model stays close to reference. High beta = strong preference enforcement, risk of reward hacking / mode collapse.

**Data requirements:** ~1K-10K preference pairs for meaningful alignment shifts. Quality of annotation matters more than quantity—a noisy pair trains the model toward the wrong objective.

---

## Beat 4: Use It

**GTM Redirect:** Zone 02 (AI-Augmented Content). Specifically: aligning a model to prefer GTM-team-approved outreach copy, sales email variants, or landing page language over generic alternatives.

**Practical setup:**
- Exercise hook (easy): Format an existing GTM copy dataset (A/B test winners vs. losers) into DPO-compatible JSONL and verify the schema.
- Exercise hook (medium): Run DPO training with a small open model (e.g., Qwen2.5-0.5B) on a synthetic preference dataset of outreach subject lines, using the `trl` library's `DPOTrainer`. Print log-probability deltas for chosen vs. rejected before and after training.
- Exercise hook (hard): Build a preference data pipeline that converts CRM email reply rates into (chosen, rejected) pairs automatically, then train and evaluate a DPO-aligned model. Measure whether the aligned model ranks high-reply variants higher than the base SFT model.

---

## Beat 5: Ship It

**GTM Redirect:** Zone 02 deployment. Shipping a DPO-aligned model means the model now has baked-in preferences—verify they're the *right* preferences before putting it in front of GTM workflows.

**Production considerations:**
- **Evaluation before deploy:** Run the aligned model on a held-out set of prompts. Score outputs with your GTM team's rubric. If the model's "chosen" outputs don't match human preference on new data, the DPO training overfit or the preference data was noisy.
- **Reference model overhead:** DPO inference doesn't need the reference model—it's only used during training. Ship the policy model alone.
- **Iterating:** New preference data requires full retraining from the SFT checkpoint (or from your last DPO checkpoint if doing iterative DPO). There's no online update mechanism.
- **Degradation detection:** Log the margin between chosen and rejected log-probs on production prompts. If the margin collapses, the model is no longer discriminating—time to retrain.

**Exercise hook (medium):** Write an evaluation script that compares base SFT model outputs vs. DPO-aligned model outputs on 20 held-out GTM prompts. Use a simple scoring function (e.g., presence of personalization tokens, length within bounds, tone classifier) to confirm alignment shifted outputs in the intended direction. Print side-by-side results.

---

## Beat 6: Evaluate

**Assessment hooks (not full quiz text):**

- **Mechanism question:** Given the DPO loss formula, explain what happens when `beta → 0` and when `beta → ∞`. What failure mode does each extreme produce?
- **Data question:** A GTM team provides 50K preference pairs where "chosen" is defined as "email sent" and "rejected" is "email not sent." Identify the confound in this labeling and propose a fix.
- **Implementation question:** Trace through a single DPO training step with concrete numbers—given log-probabilities for chosen/rejected under reference and policy models, compute the loss manually.
- **Comparison question:** Your org has a working RLHF pipeline with a trained reward model. State one scenario where switching to DPO is clearly worse and one where it's clearly better.
- **Production question:** A DPO-aligned model deployed in a GTM workflow starts generating repetitive, low-quality outputs after two weeks. Name three hypotheses for why and how you'd test each.

---

## Learning Objectives

1. **Implement** a DPO training run using preference data and a reference model, producing observable log-probability shifts between chosen and rejected outputs.
2. **Compare** DPO to RLHF in terms of data requirements, computational cost, and failure modes.
3. **Diagnose** common DPO training failures (reward hacking, mode collapse, noisy labels) from training metrics.
4. **Configure** a preference dataset from GTM A/B test results into valid (prompt, chosen, rejected) triples.
5. **Evaluate** whether a DPO-aligned model's outputs shifted in the intended direction on held-out prompts using automated scoring criteria.