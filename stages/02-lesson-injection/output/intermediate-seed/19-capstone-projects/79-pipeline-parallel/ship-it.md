## Ship It

This CLI tool wraps the schedule simulator into a planning utility. You feed it a training config (or an enrichment config) and it outputs the recommended schedule, expected bubble fraction, and peak memory estimate. No GPU required — run this before committing compute to avoid discovering at hour 3 of a training run that your bubble fraction is 40%.

```python
import sys
import json

def analyze_config(config):
    world_size = config["world_size"]
    num_stages = config["pipeline_stages"]
    num_microbatches = config["microbatches"]
    grad_accum = config.get("gradient_accumulation_steps", 1)
    hidden_size = config.get("hidden_size", 4096)
    seq_len = config.get("seq_len", 2048)
    dtype_bytes = config.get("dtype_bytes", 2)
    
    if num_stages > world_size:
        num_stages = world_size
    
    bubble_gpipe = (num_stages - 1) / (num_microbatches + num_stages - 1)
    bubble_1f1b = bubble_gpipe
    chunks = config.get("num_virtual_chunks", 1)
    effective_mb = num_microbatches * chunks
    bubble_interleaved = (num_stages - 1) / (effective_mb + num_stages - 1)
    
    activation_per_mb = num_stages * hidden_size * seq_len * dtype_bytes
    gpipe_peak_mb = num_microbatches * activation_per_mb / num_stages
    onef1b_peak_mb = num_stages * activation_per_mb / num_stages
    interleaved_peak_mb = num_stages * chunks * activation_per_mb / num_stages
    
    if bubble_gpipe > 0.2 and num_microbatches < num_stages * 4:
        recommendation = "interleaved_1f1b"
        reason = f"Bubble fraction {bubble_gpipe:.1%} is high; virtual stages reduce it to {bubble_interleaved:.1%}"
    elif config.get("memory_constrained", False):
        recommendation = "1f1b"
        reason = f"Memory-constrained; 1F1B bounds activations to {num_stages} instead of {num_microbatches}"
    else:
        recommendation = "gpipe"
        reason = f"Bubble fraction {bubble_gpipe:.1%} is acceptable; GPipe is simplest to implement"
    
    report = {
        "config": {
            "world_size": world_size,
            "pipeline_stages": num_stages,
            "microbatches": num_microbatches,
            "gradient_accumulation_steps": grad_accum,
        },
        "bubble_fractions": {
            "gpipe": round(bubble_gpipe, 4),
            "1f1b": round(bubble_1f1b, 4),
            "interleaved_1f1b": round(bubble_interleaved, 4),
        },
        "peak_activation_memory_per_stage_gb": {
            "gpipe": round(gpipe_peak_mb / 1e9, 2),
            "1f1b": round(onef1b_peak_mb / 1e9, 2),
            "interleaved_1f1b": round(interleaved_peak_mb / 1e9, 2),
        },
        "recommended_schedule": recommendation,
        "reason": reason,
        "effective_microbatches": effective_mb,
    }
    return report

configs = [
    {
        "name": "70B model, 8 GPUs, small batches",
        "world_size": 8,
        "pipeline_stages": 8,
        "microbatches": 4,
        "gradient_accumulation_steps": 4,
        "hidden_size": 8192,
        "seq_len": 2048,
        "dtype_bytes": 2,
        "memory_constrained": True,
    },
    {
        "name": "13B model, 4 GPUs, large batches",
        "world_size": 4,
        "pipeline_stages": 4,
        "microbatches": 32,
        "gradient_accumulation_steps": 1,
        "hidden_size": 5120,
        "seq_len": 2048,
        "dtype_bytes": 2,
        "memory_constrained": False,
    },
    {
        "name": "7B model, 4 GPUs, interleaved candidate",
        "world_size": 4,
        "pipeline_stages": 4,
        "microbatches": 8,
        "gradient_accumulation_steps": 2,
        "hidden_size": 4096,
        "seq_len": 4096,
        "dtype_bytes": 2,
        "num_virtual_chunks": 2,
        "memory_constrained": False,
    },
]

for cfg in configs:
    name = cfg.pop("name")
    report = analyze_config(cfg)
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")
    print(json.dumps(report, indent=2))

enrichment_config = {
    "name": "Enrichment waterfall: 5 steps, 1000 contacts",
    "world_size": 5,
    "pipeline_stages": 5,
    "microbatches": 10,
    "gradient_accumulation_steps": 1,
    "hidden_size": 1,
    "seq_len": 1,
    "dtype_bytes": 1,
    "memory_constrained": False,
}
name = enrichment_config.pop("name")
report = analyze_config(enrichment_config)
print(f"\n{'='*60}")
print(f"  Config: {name}")
print(f"{'='*60}")
b = report["bubble_fractions"]["gpipe"]
print(f"  Bubble fraction (idle enrichment steps): {b:.1%}")
print(f"  Utilization: {1-b:.1%}")
print(f"  To reach <5% bubble, need M >= {int((5-1)/0.05 - 5 + 1)} batches")
```

The output gives you a decision before you spin up GPUs (or before you launch a 10,000-contact enrichment run). The bubble fractions tell you whether your batch count is sufficient. The memory estimates tell you whether 1F1B's activation bounding is necessary.