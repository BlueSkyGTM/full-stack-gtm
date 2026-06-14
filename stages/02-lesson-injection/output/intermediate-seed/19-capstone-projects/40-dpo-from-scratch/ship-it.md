## Ship It

Production DPO fails on noisy preferences. The loss assumes each `(prompt, chosen, rejected)` triple is a clean signal — chosen is genuinely better than rejected. In GTM data, this is rarely clean. A reply might come from an out-of-office autoreplyer. A no-reply might mean the email was good but the timing was wrong. A lead might reply to variant B for reasons unrelated to the copy (they recognized the sender's company name). Noisy preferences produce a noisy gradient, and the policy learns a noisy routing function.

Filter ambiguous pairs before training. Compute the embedding similarity between chosen and rejected completions — if they are nearly identical (cosine similarity > 0.95), the preference signal is weak and should be dropped. Compute the reply rate per variant across the full campaign — if variant A has a 3% reply rate and variant B has a 2.8% reply rate, individual pairs are nearly coin flips and you need aggregate-level preferences, not instance-level ones. Drop pairs where the "chosen" completion has fewer than a threshold number of positive signals (e.g., require at least 3 replies for a variant before treating any pair as a clean preference).

```python
import torch
import torch.nn.functional as F

def filter_preference_pairs(profiles, chosen_embeds, rejected_embeds, reply_counts, min_replies=3, max_similarity=0.95):
    kept_indices = []

    for i in range(len(profiles)):
        sim = F.cosine_similarity(chosen_embeds[i:i+1], rejected_embeds[i:i+1]).item()

        if sim > max_similarity:
            continue

        if reply_counts[i] < min_replies:
            continue

        kept_indices.append(i)

    kept_tensor = torch.tensor(kept_indices)
    print(f"Original pairs: {len(profiles)}")
    print(f"After filtering: {len(kept_indices)}")
    print(f"Filtered out: {len(profiles) - len(kept_indices)}")
    return kept_tensor

torch.manual_seed(42)
n = 100
profiles = torch.randn(n, 6)
chosen_embeds = torch.randn(n, 32)
rejected_embeds = chosen_embeds + 0.05 * torch.randn(n, 32)
rejected_embeds[:20] = chosen_embeds[:20] + 0.01 * torch.randn(20, 32)
reply_counts = torch.randint(0, 8, (n,))
reply_counts[:30] = 1

kept = filter_preference_pairs(profiles, chosen_embeds, rejected_embeds, reply_counts)
```

The reference model choice matters more than most practitioners expect. The reference anchors the KL penalty — it defines what "close to the original behavior" means. If the reference is a general-purpose base model with no copywriting fine-tuning, DPO will shift the policy toward preferences but the base quality of generated text may degrade because the reference was never good at copy to begin with. If the reference is your SFT model (the one already tuned on your best historical copy), DPO shifts are smaller and more controlled — the policy stays in the neighborhood of text that already reads well. Always use your SFT model as the reference, not the base model.

Label leakage is the silent killer. When you generate preference labels using your own model (e.g., using GPT-4 to judge which email is better), the judge's biases leak into the training data. If the judge prefers longer emails, the policy will learn to generate longer emails — not better ones. Detect this by splitting your preference data by judge: if you use multiple judges or multiple judging runs, train on one set and evaluate on another. If win-rates collapse on the held-out set, you have leakage. The fix is to use behavioral signals (actual replies, actual meetings booked) as the preference source whenever possible, and reserve model-based judging for edge cases where behavioral data is too sparse.

Evaluation requires a win-rate benchmark, not just DPO loss. DPO loss going down means the policy is fitting the training preferences — it says nothing about generalization. Sample completions from both the reference and the policy on held-out prompts, then compare: does the policy output get more replies, rank higher in human evaluation, or win more often under a separate judge model? Track win-rate as the primary metric, not DPO loss.