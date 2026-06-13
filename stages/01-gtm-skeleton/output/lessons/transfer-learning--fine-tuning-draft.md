# Transfer Learning & Fine-Tuning

## GTM Redirect Rules
- Primary cluster: Zone 1 (Enrich) — custom classifiers for ICP matching, domain-adapted entity extraction
- Secondary cluster: Zone 2 (Score) — fine-tuned models for lead prioritization
- Redirect language: "This is the mechanism behind custom intent classifiers that route leads into Clay waterfall enrichment sequences"

---

## Beat 1: Hook

**Why this matters now.** You're paying per token to ask GPT-4 "is this company a fit?" 500 times a day, and getting inconsistent labels. Transfer learning and fine-tuning are the mechanisms that let you compress that decision into a smaller, cheaper, deterministic model that runs locally or at fraction of cost. Not hypothetical — this is how production GTM systems classify and route at scale.

---

## Beat 2: Concept

**Mechanism first.** Transfer learning: a model trained on Task A has learned representations (weights) that partially solve Task B. You freeze most weights, replace or retrain the final layer(s). Fine-tuning: you unfreeze some or all weights and continue training on domain-specific data with a low learning rate. The key distinction — feature extraction (frozen backbone) vs. full fine-tuning (unfrozen backbone) — and when each fails. Introduce parameter-efficient fine-tuning (LoRA) as the practical middle ground: inject low-rank matrices instead of retraining all weights.

Key terms: backbone, head, freezing, learning rate, catastrophic forgetting, rank (r) in LoRA.

---

## Beat 3: Demo

**Working code, observable output.** Python script using `transformers` and `peft` (Hugging Face) that:
1. Loads a pre-trained text classification model
2. Freezes the backbone, trains only the classifier head on a small synthetic dataset (e.g., 50 examples: "enterprise lead" vs. "SMB lead" based on company description)
3. Prints predictions and training loss

No fill-in-the-blank. Code runs in terminal, prints classification results on held-out examples.

Exercise hooks:
- **Easy:** Run the script, modify the training data labels, observe prediction changes
- **Medium:** Switch from frozen backbone to LoRA (r=8), compare loss curves
- **Hard:** Fine-tune on your own company's ICP descriptions, evaluate precision/recall on test set

---

## Beat 4: Use It

**GTM application.** This is the mechanism for building custom lead classifiers that feed into Clay waterfall enrichment sequences. Instead of asking an LLM "is this a good fit?" per record, you fine-tune a small model on 200 labeled examples from your CRM and run inference at 10,000 records/minute.

Specific application: classify inbound leads into ICP tiers (A/B/C) based on firmographic descriptions. Output feeds directly into branching logic in a Clay table — Tier A gets full waterfall enrichment (Clearbit → Hunter → Apollo), Tier C gets discarded.

[CITATION NEEDED — concept: Clay waterfall enrichment branching logic documentation]

Exercise hooks:
- **Easy:** Label 30 companies from your CRM as fit/non-fit, format as training JSON
- **Medium:** Fine-tune a classifier on your labels, integrate predictions into a Clay formula column
- **Hard:** Build a two-stage pipeline — LoRA classifier for routing, then targeted LLM enrichment only for Tier A leads; measure cost delta vs. LLM-on-everything baseline

---

## Beat 5: Ship It

**Production deployment.** Export the fine-tuned model. Serve it via a lightweight API (FastAPI + ONNX runtime or raw `transformers` pipeline). Show the inference endpoint, latency benchmarks, and how to call it from Clay's HTTP integration or n8n webhook.

Address failure modes: label drift (your ICP changes, model doesn't), data leakage (train/test overlap), class imbalance (90% non-fit leads in real data). Monitoring: track prediction distribution over time — if your classifier suddenly predicts 80% Tier A, something broke.

Exercise hooks:
- **Easy:** Export your fine-tuned model, load it in a fresh script, confirm identical predictions
- **Medium:** Wrap inference in a FastAPI endpoint, call it via `curl`, return JSON compatible with Clay webhook format
- **Hard:** Set up prediction distribution monitoring — log daily classification counts, alert if any tier shifts >20% from rolling 7-day average

---

## Beat 6: Assess It

**Verification against learning objectives.**

Learning objectives (action verbs, testable):
1. **Explain** the difference between feature extraction (frozen backbone) and full fine-tuning, including when each causes failure
2. **Implement** LoRA-based fine-tuning on a pre-trained text classification model using synthetic or real GTM-labeled data
3. **Evaluate** fine-tuned model performance using precision, recall, and prediction distribution analysis
4. **Deploy** a fine-tuned model as an inference endpoint callable from GTM orchestration tools (Clay, n8n, or equivalent)
5. **Diagnose** common failure modes (catastrophic forgetting, label drift, class imbalance) from model output patterns

Exercise hooks:
- **Easy:** Given a confusion matrix output from the demo script, identify which failure mode (class imbalance, overfitting, underfitting) is most likely
- **Medium:** Modify the training script to use 10x more data with 90% negative class — implement weighted loss to compensate, show improved recall
- **Hard:** You have 500 labeled leads from Q1. Your Q3 lead profile has shifted. Design a retraining cadence and data collection protocol to keep the classifier current without manual relabeling of all 500 records

---

**Note:** This lesson requires a `docs/en.md` with the five objectives above before any quiz bank is written. Do not generate quiz questions without that file in place.