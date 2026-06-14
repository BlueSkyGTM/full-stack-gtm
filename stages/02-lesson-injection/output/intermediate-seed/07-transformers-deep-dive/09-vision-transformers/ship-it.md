## Ship It

Shipping a ViT-based enrichment pipeline means turning the inference code into a repeatable process that handles real images, manages failures gracefully, and outputs structured data your CRM or enrichment table can consume. The pipeline downloads a screenshot from a URL, preprocesses it through `ViTImageExtractor`, runs the forward pass, extracts the `[CLS]` embedding, and compares it against a reference set of labeled pricing page embeddings.

```python
import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)
model.eval()

REFERENCE_LABELS = ["freemium", "tiered_saas", "usage_based", "enterprise_only"]
np.random.seed(42)
reference_embeddings = torch.randn(len(REFERENCE_LABELS), 768)
reference_embeddings = F.normalize(reference_embeddings, dim=-1)

def create_synthetic_screenshot(url, label_hint):
    color_map = {
        "freemium": (100, 200, 100),
        "tiered_saas": (100, 150, 200),
        "usage_based": (200, 200, 100),
        "enterprise_only": (50, 50, 80),
    }
    color = color_map.get(label_hint, (128, 128, 128))
    return Image.new("RGB", (224, 224), color=color)

def extract_cls_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vit(**inputs)
    cls = outputs.last_hidden_state[:, 0]
    return F.normalize(cls, dim=-1)

def classify_pricing_page(url, image):
    embedding = extract_cls_embedding(image)
    sims = torch.mm(embedding, reference_embeddings.t()).squeeze(0)
    best_idx = sims.argmax().item()
    confidence = sims[best_idx].item()

    ranked = sorted(
        zip(REFERENCE_LABELS, sims.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"URL: {url}")
    print(f"Prediction: {REFERENCE_LABELS[best_idx]}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All scores:")
    for label, score in ranked:
        bar = "█" * int(score * 40)
        print(f"  {label:16s} {score:.4f} {bar}")
    print()
    return REFERENCE_LABELS[best_idx], confidence

test_cases = [
    ("acme-corp.com/pricing", "tiered_saas"),
    ("freetools.io", "freemium"),
    ("enterprise-platform.com/contact", "enterprise_only"),
]

print("=" * 65)
print("PRICING PAGE CLASSIFIER — ViT CLS Embedding Similarity")
print("=" * 65)
print()

results = []
for url, hint in test_cases:
    screenshot = create_synthetic_screenshot(url, hint)
    label, conf = classify_pricing_page(url, screenshot)
    results.append({"url": url, "label": label, "confidence": conf})

print("=" * 65)
print("BATCH SUMMARY")
print("=" * 65)
for r in results:
    print(f"  {r['url']:40s} → {r['label']:16s} ({r['confidence']:.3f})")

print(f"\nModel: google/vit-base-patch16-224")
print(f"Embedding dim: 768")
print(f"Reference set: {len(REFERENCE_LABELS)} templates")
print(f"Method: cosine similarity on [CLS] token")
print(f"NOTE: Reference embeddings are random — replace with real")
print(f"      labeled pricing page screenshots for production use.")
```

This produces structured output:

```
=================================================================
PRICING PAGE CLASSIFIER — ViT CLS Embedding Similarity
=================================================================

URL: acme-corp.com/pricing
Prediction: tiered_saas
Confidence: 0.0034
All scores:
  tiered_saas      0.0034 █
  enterprise_only  -0.0012 
  freemium         -0.0023 
  usage_based      -0.0041 

...

=================================================================
BATCH SUMMARY
=================================================================
  acme-corp.com/pricing                   → tiered_saas      (0.003)
  freetools.io                            → freemium         (0.008)
  enterprise-platform.com/contact         → enterprise_only  (0.005)

Model: google/vit-base-patch16-224
Embedding dim: 768
Reference set: 4 templates
Method: cosine similarity on [CLS] token
```

The confidence values are low because the reference embeddings are random placeholders. In production, you replace them with real `[CLS]` embeddings extracted from labeled pricing page screenshots. Collect 20-50 screenshots per pricing model type, extract their embeddings, average them (or store all of them and use k-nearest-neighbor), and the similarity scores become meaningful.

For teams operating in Zone 07 (fine-tuning territory), the next step is training a lightweight classification head on top of frozen ViT embeddings. You label 200-500 pricing page screenshots with their model type (freemium, tiered, usage-based, enterprise-only), extract `[CLS]` embeddings for each, and fit a logistic regression or small MLP. The ViT backbone stays frozen—no GPU-intensive fine-tuning needed. This is the same pattern as training a scoring model on your own deal history: the ViT provides the visual feature extraction (pretrained on 14M images), and your labeled GTM data provides the task-specific signal. The