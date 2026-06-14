## Ship It

Putting weight loading into production means wrapping it in three guarantees: the checkpoint is authentic (not tampered), every parameter is accounted for (no silent partial loads), and the model produces deterministic output on known inputs before it serves traffic. The safetensors format handles the first concern by eliminating the deserialization exploit surface — there's no executable code in the file to tamper with. The second and third concerns require explicit validation code that runs at startup and fails loudly if anything is off.

For a RAG system serving outbound copy generation, the startup validation should load the embedding model, run a canonical query through it, and check that the embedding matches a stored reference vector within a tolerance. If someone swaps the checkpoint, changes the architecture, or a partial load slips through, the validation catches it before the model touches a prospect's data. This is the same pattern as a database migration check — verify the schema before serving requests.

```python
import torch
import torch.nn as nn
import json
import os
import tempfile
import hashlib

class WeightLoader:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.validation_input = None
        self.validation_output = None

    def capture_reference(self, input_tensor: torch.Tensor) -> None:
        self.validation_input = input_tensor
        self.model.eval()
        with torch.no_grad():
            self.validation_output = self.model(input_tensor).clone()

    def load_and_validate(
        self,
        checkpoint_path: str,
        expected_keys: list,
        rename_rules: dict = None,
        output_tolerance: float = 1e-5
    ) -> bool:
        rename_rules = rename_rules or {}
        
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                sd = load_file(checkpoint_path)
            except ImportError:
                raise RuntimeError("safetensors required to load .safetensors files")
        else:
            sd = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
        
        model_sd = self.model.state_dict()
        model_keys = set(model_sd.keys())
        
        applied = {}
        for ckpt_key, tensor in sd.items():
            target = ckpt_key
            for old, new in rename_rules.items():
                target = target.replace(old, new)
            if target in model_keys:
                if model_sd[target].shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {target}: "
                        f"model={tuple(model_sd[target].shape)} "
                        f"checkpoint={tuple(tensor.shape)}"
                    )
                applied[target] = tensor
            else:
                print(f"WARN: checkpoint key '{ckpt_key}' has no model counterpart")
        
        missing = set(expected_keys) - set(applied.keys())
        if missing:
            raise ValueError(f"Missing required parameters after load: {missing}")
        
        for key, tensor in applied.items():
            model_sd[key].copy_(tensor.to(self.device))
        
        if self.validation_input is not None:
            self.model.eval()
            with torch.no_grad():
                new_output = self.model(self.validation_input)
            max_diff = (self.validation_output - new_output).abs().max().item()
            if max_diff > output_tolerance:
                raise RuntimeError(
                    f"Output validation failed: max diff {max_diff} > tolerance {output_tolerance}"
                )
            print(f"Validation passed: max output diff = {max_diff:.2e}")
        
        print(f"Loaded {len(applied)}/{len(expected_keys)} parameters successfully")
        return True

class OutreachEmbedder(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_layer1 = nn.Linear(embed_dim, hidden_dim)
        self.encoder_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = torch.relu(self.encoder_layer1(x))
        x = torch.relu(self.encoder_layer2(x))
        return self.output_projection(x)

torch.manual_seed(999)
production_model = OutreachEmbedder()
reference_input = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])

with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "outreach_embedder.pt")
    torch.save(production_model.state_dict(), ckpt_path)
    
    expected_keys = list(production_model.state_dict().keys())
    
    print("=== Deploying fresh model with saved weights ===")
    deploy_model = OutreachEmbedder()
    
    fresh_input = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    deploy_model.eval()
    with torch.no_grad():
        pre_load_output = deploy_model(fresh_input)
    print(f"Pre-load output sample: {pre_load_output[0, :3].tolist()}")
    
    loader = WeightLoader(deploy_model)
    loader.capture_reference(production_model(reference_input) if False else reference_input)
    
    production_model.eval()
    with torch.no_grad():
        ref_out = production_model(reference_input)
    loader.validation_output = ref_out.clone()
    
    success = loader.load_and_validate(
        checkpoint_path=ckpt_path,
        expected_keys=expected_keys,
    )
    
    with torch.no_grad():
        post_load_output = deploy_model(fresh_input)
    print(f"Post-load output sample: {post_load_output[0, :3].tolist()}")
    
    output_changed = not torch.allclose(pre_load_output, post_load_output, atol=1e-5)
    print(f"Output changed after load: {output_changed}")
    print(f"Deployment {'READY' if success else 'BLOCKED'}")
    
    print("\n=== Testing deployment with tampered checkpoint (missing layer) ===")
    tampered_path = os.path.join(tmpdir, "tampered.pt")
    tampered_sd = {k: v for k, v in production_model.state_dict().items() 
                   if "output_projection" not in k}
    torch.save(tampered_sd, tampered_path)
    
    try:
        tampered_model = OutreachEmbedder()
        tampered_loader = WeightLoader(tampered_model)
        tampered_loader.validation_output = ref_out.clone()
        tampered_loader.load_and_validate(
            checkpoint_path=tampered_path,
            expected_keys=expected_keys,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Deployment BLOCKED: {e}")
```

The first deployment passes validation because every parameter loads correctly and the output matches the reference. The tampered checkpoint triggers `ValueError: Missing required parameters after load` — the `output_projection` keys are absent and the loader refuses to proceed. In production, that exception halts startup and surfaces in your monitoring before any outreach copy gets generated from a broken embedding model.