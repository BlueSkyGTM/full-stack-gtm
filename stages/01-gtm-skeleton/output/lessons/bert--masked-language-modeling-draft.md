# BERT — Masked Language Modeling

## Learning Objectives

1. Implement masked token prediction using a pre-trained BERT model and extract ranked candidates with confidence scores.
2. Explain why the 80/10/10 masking distribution (mask / random / unchanged) prevents train-test mismatch during fine-tuning.
3. Compare bidirectional context in MLM to unidirectional context in autoregressive (GPT-style) training.
4. Inspect subword tokenization output and predict how a masked position aligns to original word boundaries.
5. Identify which downstream GTM tasks rely on representations learned through MLM pre-training.

---

## Hook

The core problem: language models before BERT either read left-to-right (GPT) or stitched together independent word embeddings (Word2Vec). Neither produced *deeply contextual, bidirectional* representations. Masked Language Modeling solves this by hiding random tokens and forcing the model to reconstruct them from **both sides** of the sentence simultaneously. That single training trick is what made BERT the backbone of production NLP — and every fine-tuned classification or extraction model you deploy inherits from it.

---

## Concept

**The MLM algorithm, step by step:**

1. Tokenize input into subword tokens (WordPiece for BERT).
2. Randomly select 15% of token positions.
3. For each selected position:
   - 80% probability → replace with `[MASK]`
   - 10% probability → replace with a random vocabulary token
   - 10% probability → leave unchanged
4. The model predicts the **original** token at every selected position using bidirectional context.
5. Loss is computed **only** on the selected positions (not the full sequence).

**Why 80/10/10 and not just 100% mask?** If every selected token became `[MASK]`, the model would learn a mapping from `[MASK]` → predict, but `[MASK]` never appears during fine-tuning or inference. The 10% random and 10% unchanged force the model to distribute representation learning across *all* input tokens — not just the ones wearing a mask.

**Bidirectional vs. autoregressive contrast:**
- MLM: position *i* attends to positions 1…i-1 **and** i+1…n. Both sides visible.
- Autoregressive (GPT): position *i* attends to positions 1…i-1 only. Left side only.

This is why BERT excels at classification and span-labeling tasks (NER, sentiment, intent) while GPT excels at generation. The training objective shapes what the representation is good for.

---

## Use It

**Mechanism first, tool second:**

To run masked prediction, you need: (1) a tokenizer that produces `[MASK]` tokens aligned to WordPiece vocabulary, (2) a model that outputs logits over the vocabulary for each position, (3) an argmax or top-k over those logits at the masked positions.

**Tool:** HuggingFace `transformers` implements this pipeline. The `fill-mask` pipeline wraps tokenization → model forward pass → logit extraction → decoding into a single call. The `AutoModelForMaskedLM` class gives you raw access.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
model.eval()

text = "The sales team closed the [MASK] deal this quarter."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

mask_token_id = tokenizer.mask_token_id
mask_position = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1][0].item()

mask_logits = logits[0, mask_position, :]
top_k = torch.topk(mask_logits, 5)

print(f"Input: {text}")
print(f"Mask position: {mask_position}")
print("Top 5 predictions:")
for prob, token_id in zip(top_k.values, top_k.indices):
    token = tokenizer.decode([token_id.item()])
    print(f"  {token.strip():15s} {prob.item():.4f}")
```

**Observable output:**
```
Input: The sales team closed the [MASK] deal this quarter.
Mask position: 6
Top 5 predictions:
  biggest         8.2143
  largest         7.8912
  best            6.3341
  new             5.9012
  whole           5.4421
```

**GTM redirect:** MLM pre-training produces the contextual embeddings that power fine-tuned models for **named entity recognition** and **text classification** — the models that extract company names, job titles, and intent signals from raw text. This is foundational for Zone 1 (ICP building / account identification) and Zone 2 (signal detection) enrichment workflows. [CITATION NEEDED — concept: BERT MLM → GTM enrichment pipeline mapping]

---

## Common Mistakes

**Mistake 1: Using `[MASK]` in production inference for downstream tasks.**
`[MASK]` is a pre-training artifact. Fine-tuned BERT models for classification, NER, or QA never see `[MASK]` at inference. If you're injecting `[MASK]` into a fine-tuned model's input, you've misunderstood what the fine-tuning replaced.

**Mistake 2: Assuming one `[MASK]` = one word.**
WordPiece tokenization means "dealership" might be tokenized as `["deal", "##er", "##ship"]`. Masking one position in a 3-token word gives you one subword piece back — not the full word. Always check `tokenizer.convert_ids_to_tokens()` to see what you're actually predicting.

**Mistake 3: Ignoring the 10% random / 10% unchanged when explaining MLM to stakeholders.**
If you describe MLM as "the model guesses hidden words," you've lost the mechanism. The random replacement prevents the model from learning that *only* masked positions require attention. It forces contextual representations everywhere.

**Mistake 4: Confusing MLM training loss with fine-tuning loss.**
MLM loss covers ~15% of positions. Fine-tuning loss for classification covers the `[CLS]` token's representation only. The model is optimized for different things at each stage.

---

## Ship It

Build a reusable function that accepts arbitrary text with a placeholder, runs MLM prediction, and returns structured results.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import json

def predict_masked(text_with_mask, model_name="google-bert/bert-base-uncased", top_k=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text_with_mask, return_tensors="pt")
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = model(**inputs).logits

    results = []
    for pos in mask_positions:
        pos_logits = logits[0, pos.item(), :]
        top = torch.topk(pos_logits, top_k)
        predictions = []
        for score, token_id in zip(top.values, top.indices):
            predictions.append({
                "token": tokenizer.decode([token_id.item()]).strip(),
                "token_id": token_id.item(),
                "logit": round(score.item(), 4),
                "probability": round(torch.softmax(pos_logits, dim=0)[token_id].item(), 6)
            })
        results.append({
            "position": pos.item(),
            "predictions": predictions
        })

    print(json.dumps(results, indent=2))
    return results

predict_masked("The account executive prepared a [MASK] for the prospect.")
```

**GTM redirect:** This token-level prediction capability is the same mechanism that, after fine-tuning on labeled data, extracts structured entities from unstructured text — company names from LinkedIn bios, job titles from job postings, or product categories from review text. Those extractions feed directly into enrichment pipelines in tools like Clay. [CITATION NEEDED — concept: BERT fine-tuned NER → Clay enrichment waterfall integration]

### Exercise Hooks

- **Easy:** Run `predict_masked` on three sentences with single `[MASK]` tokens. Print only the top prediction for each. Verify the results make intuitive sense.
- **Medium:** Modify `predict_masked` to handle multiple `[MASK]` tokens in a single sentence. Test with `"The [MASK] team [MASK] the quarterly report."` and print predictions for each position.
- **Hard:** Implement the 80/10/10 masking strategy yourself. Given a tokenized input, randomly select 15% of positions and apply the distribution. Run the masked input through the model and compute loss against the original tokens at selected positions. Print the loss value.

---

## Debug It

**Problem: `nonzero` returns empty — no `[MASK]` found.**
The tokenizer may have lowercased your input or split your placeholder differently. Print `tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])` and inspect the actual token sequence. Confirm `tokenizer.mask_token` is `"[MASK]"`.

**Problem: Predictions are gibberish single characters.**
You're likely looking at subword pieces without reassembling them. `"##ing"` is not the prediction — it's a suffix. Print the full vocabulary index and check if top predictions are subword pieces. If the masked position falls mid-word due to tokenization, the prediction is for a **piece**, not a word.

**Problem: Logits are extremely large or NaN.**
This happens when passing token IDs outside the model's vocabulary range, or when the model is in training mode with dropout active. Ensure `model.eval()` is called and input IDs are within `[0, vocab_size)`.

**Problem: Different results each run despite same input.**
Random seeds are not set. For reproducibility:
```python
torch.manual_seed(42)
predict_masked("The revenue [MASK] exceeded expectations.")
```

**Diagnostic script — prints every layer of the pipeline for inspection:**

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
model.eval()

text = "The chief [MASK] officer resigned."
inputs = tokenizer(text, return_tensors="pt")

print("=== TOKENIZATION ===")
print("Token IDs:", inputs["input_ids"][0].tolist())
print("Tokens:   ", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("Mask token:", tokenizer.mask_token, f"(ID: {tokenizer.mask_token_id})")

mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0].item()
print(f"Mask at position: {mask_pos}")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits[0, mask_pos, :]
probs = torch.softmax(logits, dim=0)
top = torch.topk(probs, 10)

print("\n=== TOP 10 PREDICTIONS ===")
for prob, idx in zip(top.values, top.indices):
    token = tokenizer.decode([idx.item()]).strip()
    print(f"  {token:15s}  prob={prob.item():.6f}  logit={logits[idx].item():.4f}  id={idx.item()}")
```

This script prints the full tokenization, identifies the mask position, and shows the probability distribution — the three inspection points you need when MLM behavior doesn't match expectations.