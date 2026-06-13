# Loading Pretrained Weights

## Hook

You've downloaded a model checkpoint. Now what? A `.bin` file is just bytes until you map those bytes to the right parameters in the right layers in the right order. This lesson breaks the mechanism of weight loading: serialization formats, state dict key matching, partial loading, and the failure modes that silently produce garbage output.

## Concept

**Serialization formats and the state dict abstraction.** PyTorch serializes weight tensors as ordered dictionaries (`state_dict`) mapping parameter names to tensor values. The loading problem reduces to: match keys from the checkpoint to keys in the model, handle mismatches, and place tensors on the correct device. We cover `torch.load` (pickle-based), `safetensors` (memory-mapped, no deserialization exploit surface), and the Hugging Face `from_pretrained` method which implements a waterfall over multiple formats. Key mismatch patterns: renamed layers (`model.layer.0` vs `model.layers.0`), missing keys, unexpected keys, and shape mismatches. Each produces a different diagnostic message and requires a different fix.

```python
import torch
import torch.nn as nn
import tempfile
import os

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

x = torch.randn(1, 4)
print("Forward pass output:", model(x))

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "checkpoint.pt")
    torch.save(model.state_dict(), path)
    
    loaded_state_dict = torch.load(path, weights_only=True)
    
    fresh_model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    
    result = fresh_model.load_state_dict(loaded_state_dict, strict=True)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
    print("Loaded model output:", fresh_model(x))
    print("Outputs match:", torch.equal(model(x), fresh_model(x)))
```

## Use It

**Load a real model checkpoint with key mismatch handling.** Practitioners frequently encounter checkpoints that don't align perfectly with model definitions — fine-tuned adapters, renamed architectures, or partial saves. We implement `load_state_dict` with `strict=False`, then manually inspect and remap mismatched keys.

GTM redirect: This is foundational for Zone III (Enrichment) and Zone IV (Engagement). Any model powering lead scoring, intent classification, or personalized outreach starts with correctly loading its weights. A silent key mismatch in an enrichment model produces wrong enrichment data — no error, just garbage into your CRM.

```python
import torch
import torch.nn as nn

class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.classifier = nn.Linear(20, 3)
    
    def forward(self, x):
        return self.classifier(torch.relu(self.encoder(x)))

class RenamedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(10, 20)
        self.head = nn.Linear(20, 3)
    
    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))

original = OriginalModel()
state_dict = original.state_dict()
print("Original keys:", list(state_dict.keys()))

renamed = RenamedModel()
result = renamed.load_state_dict(state_dict, strict=False)
print("Missing keys (need manual mapping):", result.missing_keys)
print("Unexpected keys (need manual mapping):", result.unexpected_keys)

key_map = {
    "encoder.weight": "backbone.weight",
    "encoder.bias": "backbone.bias",
    "classifier.weight": "head.weight",
    "classifier.bias": "head.bias",
}

mapped_state_dict = {key_map[k]: v for k, v in state_dict.items()}
result = renamed.load_state_dict(mapped_state_dict, strict=True)
print("After remapping — missing keys:", result.missing_keys)
print("After remapping — unexpected keys:", result.unexpected_keys)

x = torch.randn(1, 10)
original.eval()
renamed.eval()
print("Outputs match:", torch.allclose(original(x), renamed(x)))
```

**Exercise hooks:**
- Easy: Save and load a state dict for a two-layer model, confirm outputs match.
- Medium: Given a checkpoint with renamed keys, write a mapping dict that produces `strict=True` success.
- Hard: Load only the encoder weights from a checkpoint into a model that has additional layers; verify the classifier layer was randomly initialized.

## Challenge

**Diagnose and fix three broken checkpoints.** Each broken checkpoint exhibits a different failure mode: key mismatch, shape mismatch, and dtype mismatch. We write diagnostic code for each and implement targeted fixes.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

print("=== Broken Checkpoint 1: Key Mismatch ===")
broken_keys = {"layers.0.weight": torch.randn(8, 4), "layers.0.bias": torch.randn(8)}
result = model.load_state_dict(broken_keys, strict=False)
print("Missing:", result.missing_keys)
print("Unexpected:", result.unexpected_keys)

print("\n=== Broken Checkpoint 2: Shape Mismatch ===")
wrong_shape = {"0.weight": torch.randn(8, 6), "0.bias": torch.randn(8)}
try:
    model.load_state_dict(wrong_shape, strict=False)
except RuntimeError as e:
    print("Shape mismatch error:", str(e)[:100])

print("\n=== Broken Checkpoint 3: Wrong dtype ==="
      )
wrong_dtype = {"0.weight": model[0].weight.to(torch.float16), "0.bias": model[0].bias.to(torch.float16)}
try:
    result = model.load_state_dict(wrong_dtype, strict=False)
    print("Loaded with wrong dtype, checking:")
    print("Model dtype:", model[0].weight.dtype)
except Exception as e:
    print("Error:", e)
```

**Exercise hooks:**
- Easy: Identify which failure mode each of three error messages indicates.
- Medium: Write a `safe_load` function that catches all three failure modes and returns a structured diagnostic.
- Hard: Implement a `partial_load` that loads matching keys, logs mismatches, and leaves non-matching layers at their initialized values.

## Ship It

**Implement a production-grade checkpoint loader with validation.** Build a `ModelLoader` class that: (1) detects format from file extension, (2) handles key remapping via a config, (3) validates shapes and dtypes before loading, (4) provides a loading report. This is the pattern used in any deployment pipeline.

GTM redirect: In a GTM stack, this maps to how enrichment models and classification models are loaded in production inference endpoints. [CITATION NEEDED — concept: production model serving patterns in GTM enrichment pipelines]. The loader is foundational for Zone III (Enrichment) — every enrichment API call routes through a model whose weights were loaded by this mechanism.

```python
import torch
import torch.nn as nn
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class LoadReport:
    loaded_keys: List[str] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[str] = field(default_factory=list)
    remapped_keys: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def summary(self):
        print(f"Loaded: {len(self.loaded_keys)} keys")
        print(f"Missing: {len(self.missing_keys)} keys")
        print(f"Unexpected: {len(self.unexpected_keys)} keys")
        print(f"Remapped: {len(self.remapped_keys)} keys")
        print(f"Errors: {len(self.errors)}")
        for e in self.errors:
            print(f"  - {e}")

class ModelLoader:
    def __init__(self, key_mapping: Optional[Dict[str, str]] = None):
        self.key_mapping = key_mapping or {}
    
    def validate_shapes(self, model_state: Dict, checkpoint: Dict) -> List[str]:
        errors = []
        for key in model_state:
            if key in checkpoint and checkpoint[key].shape != model_state[key].shape:
                errors.append(f"Shape mismatch for {key}: model={model_state[key].shape}, checkpoint={checkpoint[key].shape}")
        return errors
    
    def load(self, model: nn.Module, checkpoint: Dict, strict: bool = False) -> LoadReport:
        report = LoadReport()
        
        checkpoint = {self.key_mapping.get(k, k): v for k, v in checkpoint.items()}
        report.remapped_keys = {old: new for old, new in self.key_mapping.items() if old in checkpoint}
        
        model_state = dict(model.named_parameters())
        shape_errors = self.validate_shapes(model_state, checkpoint)
        report.errors.extend(shape_errors)
        
        if shape_errors and strict:
            return report
        
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state and not any(k in e for e in shape_errors)}
        
        result = model.load_state_dict(filtered_checkpoint, strict=False)
        report.loaded_keys = list(filtered_checkpoint.keys())
        report.missing_keys = result.missing_keys
        report.unexpected_keys = result.unexpected_keys
        
        return report

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

checkpoint = {
    "encoder.weight": torch.randn(8, 4),
    "encoder.bias": torch.randn(8),
    "head.weight": torch.randn(2, 8),
    "head.bias": torch.randn(2),
}

mapping = {"encoder.weight": "0.weight", "encoder.bias": "0.bias", "head.weight": "2.weight", "head.bias": "2.bias"}

loader = ModelLoader(key_mapping=mapping)
report = loader.load(model, checkpoint)
report.summary()
```

**Exercise hooks:**
- Easy: Extend `ModelLoader` to detect `.pt` vs `.safetensors` from filename.
- Medium: Add dtype validation — if checkpoint has `float16` but model expects `float32`, convert automatically and log it.
- Hard: Add layer freezing — after loading, freeze all loaded layers (set `requires_grad=False`) and log which layers remain trainable.

## Dig Deeper

- **SafeTensors format:** Read the specification at huggingface.co/docs/safetensors. Compare file sizes and load times against pickle-based `.pt` files for a 100M parameter model.
- **Hugging Face `from_pretrained` waterfall:** Trace the source code of `PreTrainedModel.from_pretrained` to see how it tries `.safetensors`, then `.bin`, then falls back to sharded checkpoints.
- **Sharded loading:** For models exceeding GPU memory, implement sequential shard loading where each shard's weights are loaded, transferred, and freed before the next.
- **GGUF format:** Used by llama.cpp. Research how quantized weights are packed and what the loading contract looks like compared to PyTorch state dicts.

**Further reading:**
- PyTorch `torch.save` / `torch.load` documentation — understand the `pickle` protocol and why `weights_only=True` exists
- safetensors spec: https://huggingface.co/docs/safetensors
- Hugging Face `modeling_utils.py` source for `load_pretrained_model` — the actual production implementation of the patterns above