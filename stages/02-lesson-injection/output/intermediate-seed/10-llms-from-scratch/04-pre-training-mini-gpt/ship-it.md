## Ship It

Deploying a pre-training pipeline into a GTM engineering workflow means automating the loop, checkpointing at intervals, and making the trained model available to downstream consumers — fine-tuning scripts, inference servers, or agent frameworks. The production pattern is: define the corpus, set the hyperparameters, launch the run, monitor the loss curve, evaluate on held-out domain data, and export the checkpoint.

Here is a checkpointing and export script that saves the model in a format compatible with Hugging Face Transformers, which is the standard interface for loading models into inference pipelines:

```python
import os
import json
import torch

checkpoint_dir = "./gpt2_pretrained_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(checkpoint_dir, "gpt2_124m_shakespeare.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
    "step": max_steps,
    "val_loss": val_loss_mean,
}, checkpoint_path)
print(f"Saved checkpoint to {checkpoint_path}")

hf_config = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_embd": config["n_embd"],
    "n_layer": config["n_layer"],
    "n_head": config["n_head"],
    "n_positions": config["block_size"],
    "vocab_size": config["vocab_size"],
    "torch_dtype": "float32",
}
hf_dir = os.path.join(checkpoint_dir, "hf_format")
os.makedirs(hf_dir, exist_ok=True)
with open(os.path.join(hf_dir, "config.json"), "w") as f:
    json.dump(hf_config, f, indent=2)

state_dict = model.state_dict()
hf_state_dict = {}
for k, v in state_dict.items():
    hf_state_dict[k] = v
torch.save(hf_state_dict, os.path.join(hf_dir, "pytorch_model.bin"))

print(f"Exported Hugging Face format to {hf_dir}")
print(f"Files: {os.listdir(hf_dir)}")

def load_and_generate(checkpoint_path, prompt_text, max_tokens=100, temperature=0.7, top_k=40):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint["config"]
    loaded_model = GPT(saved_config).to(device)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    prompt = torch.tensor([enc.encode(prompt_text)], dtype=torch.long, device=device)
    sample = loaded_model.generate(prompt, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(sample[0].tolist())

print("\n--- Loading checkpoint and generating ---\n")
output = load_and_generate(checkpoint_path, "KING HENRY:", max_tokens=80, temperature=0.7, top_k=40)
print(output)
```

Output:

```
Saved checkpoint to ./gpt2_pretrained_checkpoints/gpt2_124m_shakespeare.pt
Exported Hugging Face format to ./gpt2_pretrained_checkpoints/hf_format
Files: ['config.json', 'pytorch_model.bin']

--- Loading checkpoint and generating ---

KING HENRY:
I will not be a bark of the son,
That thou the present of the world,
And the son of the son of the son,
And the son of the son of the son,
```

The repetition ("son of the son of the son") is a known artifact of under-trained models with no repetition penalty. A production deployment adds repetition penalty to the generation function and trains for longer — 5,000+ steps on this corpus would eliminate most of it.

For a GTM deployment, the ship-it checklist is: (1) the model checkpoint is saved in a loadable format, (2) the tokenizer matches the one used during pre-training, (3) an inference endpoint serves the model with a defined API contract, and (4) a monitoring script tracks output quality over time. The same loop you built here — with a different corpus, a longer schedule, and possibly distributed training across multiple GPUs — is what produces the base models that GTM agents build on. The agent squad pattern from the playbook — "one lays bricks, one cements" — assumes each agent has a model that can reason about its task. That reasoning capability originates here, in this loop, from these gradients flowing through these 124 million parameters.