## Ship It

The production configuration for DP fine-tuning of an LLM in 2025 is LoRA + DP-SGD via Opacus (PyTorch) or TensorFlow Privacy. You freeze the base model weights, apply low-rank adapters to attention projections, and run DP-SGD only on the adapter parameters. This reduces the per-example gradient memory cost from "all transformer parameters" to "just the adapter matrices," making DP fine-tuning feasible on a single A100 for 7B-parameter models. [CITATION NEEDED — concept: LoRA + DP-SGD memory cost reduction factor vs full DP fine-tuning]

For the GTM pipeline, this means you can fine-tune on customer conversation data extracted from Salesforce, Gong, or Zendesk, generate the account insights and personalized copy that Zone 18's ABM personalization chains require, and ship with a defensible privacy guarantee. The deployment checklist is: (1) compute your target ε based on legal/security requirements, (2) derive σ from the accountant formula, (3) run DP-SGD with LoRA, (4) log the achieved ε to your model registry, and (5) validate with a canary insertion test — plant a fake "canary" ticket in the training data and verify the model cannot reproduce it when prompted.

```python
import math
import json

def build_dp_finetuning_config(dataset_size, batch_size, epochs, target_epsilon,
                                delta=1e-5, base_model="meta-llama/Llama-2-7b-hf"):
    steps = (dataset_size // batch_size) * epochs
    q = batch_size / dataset_size
    sigma = q * math.sqrt(steps * math.log(1.0 / delta)) / target_epsilon
    sigma = max(sigma, 0.5)
    achieved_eps = q * math.sqrt(steps * math.log(1.0 / delta)) / sigma

    config = {
        "base_model": base_model,
        "method": "LoRA-DP-SGD",
        "dataset": {
            "size": dataset_size,
            "source": "crm_tickets + gong_transcripts",
            "pii_categories": ["name", "email", "phone", "account_id"]
        },
        "dp_config": {
            "target_epsilon": target_epsilon,
            "delta": delta,
            "noise_multiplier": round(sigma, 2),
            "max_grad_norm": 1.0,
            "achieved_epsilon": round(achieved_eps, 4),
            "accountant": "moments"
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "steps": steps,
            "sampling_rate": round(q, 6),
            "learning_rate": 1e-4,
            "optimizer": "adamw"
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        },
        "framework": {
            "name": "opacus",
            "version": ">=1.4",
            "backend": "pytorch"
        },
        "audit": {
            "canary_test": "insert 10 canary records, verify extraction fails",
            "mia_threshold": "membership inference AUC < 0.55",
            "log_epsilon_to_registry": True
        }
    }
    return config

config = build_dp_finetuning_config(
    dataset_size=50000,
    batch_size=256,
    epochs=3,
    target_epsilon=4.0
)

print("=== Production DP Fine-Tuning Config ===")
print(json.dumps(config, indent=2))

print(f"\n=== Deployment Verification ===")
print(f"Target ε:     {config['dp_config']['target_epsilon']}")
print(f"Achieved ε:   {config['dp_config']['achieved_epsilon']}")
print(f"σ required:   {config['dp_config']['noise_multiplier']}")
print(f"Influence cap: e^ε = {math.exp(config['dp_config']['achieved_epsilon']):.2f}x")
print(f"Total steps:  {config['training']['steps']}")
print(f"\nNext: run Opacus with this σ, then insert canaries and verify extraction fails.")
```

One deployment caveat worth stating explicitly: the DP guarantee applies to the fine-tuning data, not the base model's pretraining data. If the base model already memorized PII from its pretraining corpus (which it likely did), DP fine-tuning does not retroactively protect