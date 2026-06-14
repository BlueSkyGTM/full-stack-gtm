## Ship It

Production image classifiers fail on distribution shift. ResNet50 trained on ImageNet has never seen a B2B SaaS pricing page, a CRUD admin panel, or a fintech dashboard. When you feed it those images, it will confidently map them to the nearest ImageNet category — "web site" is not a class, so you get "digital clock" or "notebook" or whatever texture pattern happens to activate most strongly. The softmax output looks valid (it sums to 1.0) but the labels are meaningless.

The fix is domain-specific calibration. Collect 200-500 images from your actual target distribution (screenshots of company homepages, logo crops, product images), label them with your actual categories, and measure the model's confidence distribution on that held-out set. If the 95th percentile confidence for correct predictions is 0.72, your acceptance threshold should be 0.72 — not the 0.50 default or the 0.90 you might assume from ImageNet benchmarks. Log per-class confidence distributions over time using a simple histogram; when the median confidence for a class drops 10-15% below its calibration baseline, the input distribution has drifted and retraining is due.

For low-latency enrichment — classifying a screenshot within a webhook response, for instance — loading PyTorch at inference time adds 2-5 seconds of startup and 1.5GB of RAM. Export the model to ONNX format and serve it via a lightweight inference server (ONNX Runtime, Triton). The convolutional operations execute identically; the startup cost drops to under 200ms. The code below demonstrates the ONNX export and a confidence-distribution diagnostic:

```python
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import urllib.request

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = "resnet50.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print(f"Exported ONNX model: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")

import os

image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
urllib.request.urlretrieve(image_url, "sample.jpg")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("sample.jpg").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
probs = torch.nn.functional.softmax(logits[0], dim=0)

sorted_probs, sorted_idx = torch.sort(probs, descending=True)
print("\nCONFIDENCE DISTRIBUTION DIAGNOSTIC")
print("-" * 55)
print(f"  Top-1 prob:    {sorted_probs[0].item():.4f}")
print(f"  Top-5 sum:     {sorted_probs[:5].sum().item():.4f}")
print(f"  Top-10 sum:    {sorted_probs[:10].sum().item():.4f}")
print(f"  Median prob:   {probs.median().item():.6f}")
print(f"  Mean prob:     {probs.mean().item():.6f}  (baseline: {1/1000:.6f})")
print(f"  Entropy:       {(-probs * probs.log()).sum().item():.4f} bits")
print(f"  Effective k:   {(1.0 / (probs ** 2).sum()).item():.1f} of 1000 classes")
print("-" * 55)
print("  DIAGNOSIS:")
if sorted_probs[0] > 0.9 and sorted_probs[1] < 0.05:
    print("    HIGH CONFIDENCE — single dominant class. Safe to auto-route.")
elif sorted_probs[0] > 0.5:
    print("    MODERATE — top class leads but runner-up is close. Consider context.")
else:
    print("    LOW CONFIDENCE — distribution is flat. Flag for manual review.")
```

The diagnostic metrics give you a calibration baseline. "Effective k" (the inverse of the sum of squared probabilities) tells you how many classes the model is effectively hedging across — a value of 1.0 means all probability mass is on one class; a value of 500 means the model is spreading uncertainty across half the label space. For production enrichment pipelines, track effective k over time. When it climbs, your inputs are drifting out of the model's competence zone.

The Clay waterfall pattern — Find → Enrich → Transform → Export — maps directly onto this setup [CITATION NEEDED — concept: Clay waterfall as Zone 4 enrichment pattern]. The ONNX-served classifier is the Transform node; the confidence threshold is the gate that decides whether a row proceeds to Export or loops back for manual enrichment.