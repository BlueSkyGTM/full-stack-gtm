## Ship It

A factory function turns the configuration dict into an initialized model. This is the deployment interface — you store configs as JSON, load them at startup, and spin up the right model without code changes.

```python
import json

def create_gpt(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    required = ["vocab_size", "context_length", "hidden_dim", "num_heads", "num_layers"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    if config["hidden_dim"] % config["num_heads"] != 0:
        raise ValueError(
            f"hidden_dim ({config['hidden_dim']}) must be divisible by "
            f"num_heads ({config['num_heads']})"
        )
    model = GPTModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    layer_mem_bytes = sum(
        sum(p.numel() * p.element_size() for p in block.parameters())
        for block in model.blocks
    )
    print(f"Model created: {config['num_layers']} layers, "
          f"{config['hidden_dim']} hidden dim, {config['num_heads']} heads")
    print(f"Total parameters: {param_count:,}")
    print(f"Parameter memory (fp32): {param_count * 4 / (1024**3):.2f} GB")
    print(f"Per-layer memory:        {layer_mem_bytes / config['num_layers'] / (1024**2):.2f} MB")
    return model

configs = {
    "gpt2_small.json": {
        "vocab_size": 50257, "context_length": 1024,
        "hidden_dim": 768, "num_heads": 12, "num_layers": 12
    },
    "gpt2_medium.json": {
        "vocab_size": 50257, "context_length": 1024,
        "hidden_dim": 1024, "num_heads": 16, "num_layers": 24
    },
}

for name, cfg in configs.items():
    with open(name, "w") as f:
        json.dump(cfg, f)

model = create_gpt("gpt2_small.json")
print()
model = create_gpt("gpt2_medium.json")
```

This factory validates the config, builds the model, and reports the memory footprint. The `hidden_dim % num_heads == 0` check catches the most common configuration error — attention heads must partition the hidden dimension evenly.

For text generation, you need sampling logic on top of the forward pass. The model produces logits; you convert to probabilities, apply temperature, truncate with top-k, and sample:

```python
def generate(model, input_ids, max_new_tokens=20, temperature=0.8, top_k=50):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -model.config["context_length"]:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_idx], dim=1)
    return input_ids

model = GPTModel(config)
model.eval()
input_ids = torch.randint(0, config["vocab_size"], (1, 4))
output = generate(model, input_ids, max_new_tokens=10, temperature=0.7, top_k=40)
print(f"Input length:  {input_ids.shape[1]}")
print(f"Output length: {output.shape[1]}")
print(f"New tokens:    {output.shape[1] - input_ids.shape[1]}")
```

The sliding window (`idx_cond = input_ids[:, -context_length:]`) handles sequences longer than the model's context by keeping only the most recent tokens. Temperature below 1.0 sharpens the distribution toward high-probability tokens; top-k truncates the tail before sampling to avoid rare-token noise.