# The Direct Preference Optimization Family

## GTM Redirect Rules
- Primary cluster: **Zone 4 — Outreach & Message Generation** (message quality alignment)
- Secondary: **Zone 3 — Scoring & Prioritization** (preference-based ranking)
- If the connection is thin: redirect as "foundational for custom model alignment in Zones 3–4"

---

## Beat 1: Open It

The RLHF pipeline has a reward model and a reinforcement learning loop. DPO collapses both into a single classification loss on paired preference data. This beat establishes why that collapse works and what it costs you — namely, the ability to update preferences without retraining.

---

## Beat 2: See It

Walk through the DPO derivation: start from the KL-constrained reward maximization objective, substitute the optimal policy in closed form, and arrive at the logistic loss on log-ratio differences. Show how IPO relaxes this by replacing the logistic with a squared hinge, and how KTO drops the pairing requirement entirely by using only binary "good/bad" labels with a baseline. Diagram the loss surface for each.

---

## Beat 3: Touch It

Implement the DPO loss function from scratch in ~20 lines of PyTorch. Run it on a toy paired dataset and print the loss values, confirming that chosen examples produce lower loss than rejected ones. Then swap in the IPO loss and observe the difference in gradient magnitude.

- **Exercise hook (easy):** Modify the beta hyperparameter and print how the loss landscape shifts.
- **Exercise hook (medium):** Implement KTO loss using only binary labels and compare training dynamics against DPO on the same data.

---

## Beat 4: Build It

Fine-tune a small language model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) on a preference dataset using `trl`'s `DPOTrainer`. Use a synthetic dataset of "good outreach email" vs. "bad outreach email" pairs. Log the reward margin (log-probability difference) across training steps. Then repeat with `ORPOTrainer` and compare convergence speed.

- **Exercise hook (hard):** Run both DPO and KTO on the same preference corpus. Log reward margins, generate 20 samples from each, and rank them with a judge model. Report win-rate against the SFT baseline.

---

## Beat 5: Use It

**GTM Redirect → Zone 4, Outreach & Message Generation**

When your outreach model generates variant emails, you need it to prefer the variants that historically earned replies. DPO-family methods let you align the model to your own reply-rate signal without training a separate reward model. Build a preference dataset from CRM outcomes: (email sent, replied = chosen; email sent, no reply = rejected). Fine-tune with DPO or KTO depending on whether you have clean pairs or only binary outcome labels.

[CITATION NEEDED — concept: DPO fine-tuning for outreach reply-rate optimization in production GTM workflows]

---

## Beat 6: Ship It

DPO-trained models drift when the preference distribution shifts (new ICP, new market, new quarter). Ship a pipeline that: (1) logs all generated outputs with outcomes, (2) periodically reconstructs the preference dataset, (3) triggers a DPO retraining run when the reward margin on new data drops below threshold. Quantify alignment freshness with a held-out preference eval set scored weekly.

- **Exercise hook (medium):** Write a script that loads a week's CRM outcome data, constructs preference pairs, and appends them to an existing JSONL dataset — ready for the next DPO run.

---

## Learning Objectives (draft)

1. **Derive** the DPO loss from the KL-constrained reward maximization objective and **explain** why the reward model becomes implicit.
2. **Implement** the DPO and IPO loss functions from scratch and **compare** their gradient behavior on identical data.
3. **Configure** `DPOTrainer` and `KTOTrainer` from the `trl` library to fine-tune a language model on a preference dataset.
4. **Construct** a preference dataset from GTM outcome signals (replied/not-replied) suitable for DPO-family alignment.
5. **Evaluate** alignment quality using reward margin and win-rate against an SFT baseline.