## Ship It

Deploying a DPO-trained model to production involves three decisions: checkpoint selection, beta calibration, and evaluation against GTM metrics. The first two are engineering problems. The third is where alignment meets revenue.

Checkpoint selection follows the same protocol as SFT: hold out 10-15% of preference pairs as a validation set, track DPO loss on that set across training steps, and select the checkpoint with the lowest validation loss. A rising validation loss while training loss continues to drop means the model is memorizing preference pairs rather than learning the underlying preference distribution. Stop early.

Beta calibration is DPO-specific. Start with beta=0.1 (the default in `trl`). Run training at beta=0.05, 0.1, 0.3, and 0.5. For each, measure the KL divergence between the trained policy and the reference model on a held-out prompt set. The KL divergence tells you how far the model has drifted from the SFT checkpoint. A drift that is too large means the model may have degraded on general capabilities while improving on the preference task.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_kl(model_a, model_b, prompts, tokenizer, max_length=128):
    model_a.eval()
    model_b.eval()
    total_kl = 0.0
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
            logits_a = model_a(**inputs).logits
            logits_b = model_b(**inputs).logits
            probs_a = F.log_softmax(logits_a, dim=-1)
            probs_b = F.softmax(logits_b, dim=-1)
            kl = F.kl_div(probs_a, probs_b, reduction="batchmean", log_target=False)
            total_kl += kl.item()
    return total_kl / len(prompts)

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

eval_prompts = [
    "Write a cold outreach subject line for: Series A AI startups",
    "Write a cold outreach subject line for: Mid-market logistics companies",
    "Explain what API rate limiting is",
    "Summarize the plot of Hamlet"
]

alignment_prompts = eval_prompts[:2]
general_prompts = eval_prompts[2:]

kl_alignment = compute_kl(base_model, base_model, alignment_prompts, tokenizer)
kl_general = compute_kl(base_model, base_model, general_prompts, tokenizer)

print(f"KL on alignment prompts (self): {kl_alignment:.6f}")
print(f"KL on general prompts (self):   {kl_general:.6f}")
print(f"\nReference values (self-comparison should be ~0)")
print(f"After DPO training, re-run with trained vs base model.")
print(f"Target: alignment KL > 0.1, general KL < 0.05")
print(f"If general KL > 0.3, beta is too high or data is too narrow")
```

```
KL on alignment prompts (self): 0.000000
KL on general prompts (self):   0.000000

Reference values (self-comparison should be ~0)
After DPO training, re-run with trained vs base model.
Target: alignment KL > 0.1, general KL < 0.05
If general KL > 0.3, beta is too high or data is too narrow
```

The self-comparison produces zero KL (as expected — identical distributions). After DPO training, swap the trained model in for `model_a` and check that alignment-domain KL is meaningfully above zero (the model shifted on outreach style) while general-domain KL stays low (the model still writes good explanations of API rate limiting and Hamlet summaries). If general-domain KL is high, the model has degraded on non-preference tasks. Lower beta or add general-purpose preference pairs to the training data.

The evaluation step connects alignment to GTM outcomes. Generate 100 outreach subject lines from both the base model and the DPO-trained model on a held-out set of segments. Send each set through your actual outreach pipeline — Clay enrichment, email send, open rate and reply rate tracking. The DPO model should produce measurably higher open rates if the preference data captured real conversion patterns rather than stylistic preferences. [CITATION NEEDED — concept: DPO-trained model outperforming base model on live outreach open rates]

In the multi-agent orchestration pattern from Zone 10, a DPO-aligned model serves as the content generation agent within a task squad. One agent scrapes prospects from niche directories via Claygent — LinkedIn for company data, industry-specific sources like gamedevmaps.com for game studios or legal directories for law firms. Another agent composes outreach using the DPO-trained model, which prefers the copy patterns your team validated. A third agent handles follow-up scheduling. The DPO model is not orchestrating — it is producing aligned content within the pipeline. [CITATION NEEDED — concept: DPO-aligned generation agent in multi-agent GTM squad]

The production checklist is short: validate preference data quality before training (ambiguous pairs where win rates are close to 50/50 add noise, not signal), hold out evaluation pairs, train at multiple beta values, measure KL drift on general tasks, and A/B test the deployed model against the base model on live outreach metrics. If open rates do not improve, the preference data did not capture the right signal. Go back to the data, not the algorithm.