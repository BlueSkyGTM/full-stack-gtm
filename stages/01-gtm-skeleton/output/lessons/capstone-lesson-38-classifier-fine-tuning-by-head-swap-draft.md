# Lesson 38: Classifier Fine-Tuning by Head Swap

## Learning Objectives

1. Replace the classification head of a pre-trained transformer model with a new head sized for a target label space.
2. Freeze the backbone weights and train only the new head on a labeled dataset.
3. Compare frozen-backbone training to full fine-tuning in terms of accuracy and training time.
4. Export the fine-tuned model and run inference on unseen inputs.
5. Diagnose when head-swap transfer learning fails due to domain mismatch.

---

## Beat 1: Hook

**The problem:** You have 200 labeled examples for a 5-class intent problem. Training a transformer from scratch will overfit and take hours. You need the linguistic knowledge already baked into a pre-trained model, but its output layer predicts 2 classes for sentiment—not your 5 intent classes.

**The move:** Swap the head. Keep the backbone that already understands language. Replace only the final layer. Train 1,000 parameters instead of 110 million.

---

## Beat 2: Concept

**Mechanism — what "head swap" means architecturally:**

A transformer classifier is two pieces: a backbone (embeddings + encoder layers) and a head (a linear layer mapping `[CLS]` hidden state → logits). The backbone learned general language representations during pre-training. The head is task-specific. Transfer learning by head swap exploits this separation.

```
Pre-trained Model (e.g., bert-base-uncased, 2-class sentiment)
┌──────────────────────────┐
│  Backbone (frozen)       │  ← 109M parameters, unchanged
│  Embedding + 12 layers   │
│  Output: [batch, seq, 768]│
└──────────┬───────────────┘
           │ [CLS] token → [batch, 768]
┌──────────▼───────────────┐
│  OLD Head: Linear(768, 2)│  ← DISCARD this
└──────────────────────────┘

Your Model (5-class intent)
┌──────────────────────────┐
│  Backbone (frozen)       │  ← Same weights, gradient disabled
└──────────┬───────────────┘
           │
┌──────────▼───────────────┐
│  NEW Head: Linear(768, 5)│  ← Train ONLY this (3,840 params)
└──────────────────────────┘
```

**Why freeze the backbone?** With 200 examples and 109M parameters, the model will memorize training data. Freezing constrains the optimization to 3,840 parameters—a ratio the data can support.

**When this fails:** If your domain uses specialized vocabulary (medical notes, legal filings, internal jargon) that the backbone never saw during pre-training, the frozen representations will be weak. No amount of head training fixes bad features. Symptoms: training loss drops but validation accuracy stays near random.

**Tools:** Hugging Face `transformers` implements this pattern via `AutoModelForSequenceClassification.from_pretrained()` with `num_labels` set to your target count. The library replaces the head automatically. `transformers` solves the problem of weight loading and architecture matching so you don't have to manually slice model state dicts.

---

## Beat 3: Demonstration

**Code example:** Load a pre-trained model, swap the head, freeze the backbone, train on a small synthetic dataset, and evaluate. All output printed to terminal.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random

random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
LABEL_NAMES = ["churn_risk", "upsell_opportunity", "support_request", "onboarding"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

print("=== Model Architecture (head only) ===")
print(model.classifier)
print()

total_params = sum(p.numel() for p in model.parameters())
head_params = sum(p.numel() for p in model.classifier.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Head parameters:  {head_params:,}")
print(f"Trainable ratio:  {head_params/total_params:.6f}")
print()

for param in model.distilbert.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"After freezing backbone:")
print(f"  Trainable: {trainable:,}")
print(f"  Frozen:    {frozen:,}")
print()

INTENT_DATA = [
    ("I want to cancel my subscription", 0),
    ("thinking about leaving your service", 0),
    ("can I upgrade to enterprise plan", 1),
    ("interested in adding more seats", 1),
    ("the app keeps crashing on login", 2),
    ("getting a 500 error when I save", 2),
    ("how do I set up my workspace", 3),
    ("where do I invite my team members", 3),
    ("looking for a better deal", 0),
    ("need to talk to someone about leaving", 0),
    ("what does the premium tier include", 1),
    ("we are growing fast need more capacity", 1),
    ("integration with slack is broken", 2),
    ("cannot connect my email account", 2),
    ("first time here how does this work", 3),
    ("help me configure my first project", 3),
]

class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }

dataset = IntentDataset(INTENT_DATA, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
model.train()

print("=== Training (head only) ===")
for epoch in range(15):
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d} | Loss: {total_loss:.4f} | Accuracy: {correct/total:.2%}")

print()

TEST_TEXTS = [
    "we want to move to a different provider",
    "can we add additional users to our plan",
    "getting timeout errors every morning",
    "just signed up what are the next steps",
]

model.eval()
print("=== Inference on unseen texts ===")
with torch.no_grad():
    for text in TEST_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0, pred].item()
        print(f"  '{text}'")
        print(f"    → {LABEL_NAMES[pred]} (confidence: {conf:.2%})")
```

---

## Beat 4: Guided Practice

**Easy exercise:** Change `NUM_LABELS` to 2 and create a binary "churn vs. not churn" dataset. Train and observe how the head shape changes (print `model.classifier` before and after).

**Medium exercise:** Unfreeze the backbone (`requires_grad = True` for all parameters), retrain with the same data and a lower learning rate (`1e-5`), and compare final accuracy. Document whether full fine-tuning helps or overfits with 16 examples.

**Hard exercise:** Add a 2-layer MLP head (`nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, NUM_LABELS))`) instead of the default single linear layer. Compare training convergence speed and final accuracy against the linear head.

---

## Beat 5: Use It

**GTM application — Intent routing for inbound messages:**

This is the mechanism behind building a custom intent classifier for lead routing. When a prospect fills out a contact form or sends an email, a head-swapped classifier categorizes the message into action buckets: "request demo," "support issue," "partnership inquiry," "spam."

[CITATION NEEDED — concept: GTM intent routing workflow using custom classifiers in Zone 03/04 lead processing]

In a Clay workflow, this classifier output triggers conditional branching: demo requests route to the AE assignment sequence, support issues create a Zendesk ticket, partnership inquiries notify the partnerships channel. The classifier is the decision node; the workflow is the automation.

**Deployment pattern:** Export the fine-tuned model to a hosted API endpoint (FastAPI + Hugging Face pipeline wrapper). Clay calls this endpoint via HTTP enrichment column. The response label feeds into a conditional formula column that drives the branch logic.

---

## Beat 6: Ship It

**Production checklist for a deployed head-swapped classifier:**

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

test_results = [
    classifier("need help with billing"),
    classifier("interested in enterprise pricing"),
    classifier("please cancel my account today"),
]

print("=== Production Smoke Test ===")
for result in test_results:
    print(f"  {result}")
```

**Ship criteria (minimum viable classifier):**
1. Accuracy ≥ 80% on a held-out test set of at least 50 examples per class.
2. Inference latency < 200ms per input on your deployment hardware.
3. Confidence calibration: when the model outputs > 0.9 confidence, it should be correct > 95% of the time. If not, the head learned to be overconfident—collect more training data.
4. Failure mode documented: what text inputs does the model silently misclassify? Run your 50-example test set and list every failure case. Those are your edge cases for the next training iteration.

**Next iteration:** Once you collect 1,000+ labeled examples, move from frozen backbone to full fine-tuning with a low learning rate. The head swap is the starting point, not the ceiling.

---

## GTM Redirect Rules

- **"Use It" redirect:** Zone 03/04 — inbound lead routing and intent classification. This is the classifier that powers conditional workflow branching in GTM automation.
- **"Ship It" redirect:** The model endpoint becomes an enrichment source in Clay. HTTP column calls the classifier, formula column branches on the label. This is how custom ML models plug into the Clay waterfall architecture.