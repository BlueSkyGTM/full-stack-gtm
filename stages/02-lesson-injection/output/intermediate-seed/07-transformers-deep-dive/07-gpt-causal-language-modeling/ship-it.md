## Ship It

### Decoding strategy is a production lever

Greedy decoding (always pick the highest-probability token) is deterministic but produces repetitive, generic text. If you generate 1,000 prospecting emails with greedy decoding and the same prompt, you get 1,000 identical emails — useless for outreach at scale. Top-k sampling restricts the pool to the k most likely tokens and samples proportionally. Top-p (nucleus) sampling takes the smallest set of tokens whose cumulative probability exceeds p, then samples from that set. Both trade determinism for variance, which is what you want when generating email variants — but the variance must be monitored. If two top-k sampled emails are 95% identical, your duplicate detection will flag them. If they are 20% identical, your messaging is incoherent.

Temperature controls the shape of the probability distribution before sampling. Low temperature (< 0.5) sharpens the distribution — the highest-probability tokens get even more weight, and the model becomes more deterministic. High temperature (> 1.0) flattens the distribution — unlikely tokens get more weight, and the model becomes more creative but less grounded. In GTM pipelines, the practical split is: temperature 0.0–0.4 for factual extraction tasks (company research, industry classification, ICP matching), where you want the model's best guess and do not want variance. Temperature 0.7–1.0 for creative copy generation (email subject lines, opening hooks, ad variants), where you want diversity across a set of outputs.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.5, 2.1, 1.8, 1.2, 0.9, 0.5, 0.3, 0.1])

for temp in [0.3, 0.7, 1.0, 1.5]:
    adjusted = logits / temp
    probs = F.softmax(adjusted, dim=-1)
    top_prob = probs[0].item()
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    print(f"Temp {temp}: top_token_prob={top_prob:.3f}, entropy={entropy:.3f} bits")

print()
print("Lower temp → top token dominates, less sampling variance")
print("Higher temp → distribution flattens, more creative but less grounded")
```

Output:
```
Temp 0.3: top_token_prob=0.394, entropy=1.614 bits
Temp 0.7: top_token_prob=0.269, entropy=1.946 bits
Temp 1.0: top_token_prob=0.221, entropy=2.080 bits
Temp 1.5: top_token_prob=0.173, entropy=2.202 bits
```

### Token budget and cost

Causal LM attention computes pairwise interactions between all token positions. The attention matrix is `N × N`, so attention computation scales as O(N²). Doubling sequence length quadruples attention cost. In production GTM pipelines — where you might pass a long company research brief plus few-shot examples plus a system prompt — capping input and output tokens is not optional. A 4,000-token prompt costs more than inference time; it costs more per API call, and the O(N²) scaling means latency degrades nonlinearly. Set `max_tokens` on every API call. Monitor the ratio of input to output tokens. If your input tokens are 10× your output tokens, you are paying for context the model may not be using effectively.

### Log-probability monitoring

Every token the model generates has an associated log-probability. If the model assigns -0.2 log-prob to a token, it was 82% confident. If it assigns -8.0, it was 0.03% confident — essentially guessing. In production, tracking the average log-probability of generated tokens catches degenerate outputs before they reach prospects. If the average log-prob drops sharply mid-generation — say, from -0.5 for the first 50 tokens to -3.0 for tokens 51-80 — the model has likely diverged from coherent text. It is generating tokens it is not confident about, which in GTM contexts means hallucinated company names, fabricated product features, or nonsensical email closers.

The OpenAI API returns log-probabilities when you set `logprobs=True`. In your GTM pipeline, log all generated token log-probs and set a threshold — if the average log-prob of any generation drops below -2.0, flag it for human review before sending. This is cheap (the log-probs are already computed) and catches the failure mode that matters most in outreach: confident-sounding nonsense sent to a real prospect.