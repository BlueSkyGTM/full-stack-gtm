## Ship It

Production deployment of CLIP requires three engineering decisions: image preprocessing, batch strategy, and similarity thresholding.

**Preprocessing** is non-negotiable. CLIP's image encoder expects 224×224 RGB images normalized with the mean and standard deviation from its training distribution. The `CLIPProcessor` handles resize + center-crop + normalize, but if you bypass it (for speed or custom pipelines), you must replicate the exact transforms. A mismatch in normalization shifts embeddings enough to break zero-shot accuracy silently — predictions look reasonable but confidence scores collapse.

**Batching** is where latency gets managed. GPU inference on a single image takes ~5–15ms with ViT-B/32. CPU inference takes ~150–200ms per image. For a 5,000-company enrichment run, that is the difference between 1 minute and 15 minutes. Batch the image encodings (8–32 images per forward pass) and pre-compute text embeddings once per taxonomy, not per image:

```python
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

taxonomy_labels = [
    "a screenshot of an e-commerce store",
    "a screenshot of a SaaS dashboard",
    "a screenshot of a blog",
    "a screenshot of a restaurant website",
    "a screenshot of a corporate landing page"
]

with torch.no_grad():
    text_inputs = processor(text=taxonomy_labels, return_tensors="pt", padding=True, truncation=True).to(device)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = torch.nn.functional.normalize(text_embeds, dim=1)

test_urls = [
    f"http://images.cocodataset.org/val2017/{str(i).zfill(12)}.jpg"
    for i in range(39769, 39769 + 50)
]

batch_size = 16
all_images = []
valid_meta = []

for idx, url in enumerate(test_urls):
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        all_images.append(img)
        valid_meta.append(idx)
    except Exception:
        pass

print(f"Downloaded {len(all_images)} images, skipping {len(test_urls) - len(all_images)} failures")

warmup_inputs = processor(images=all_images[:2], return_tensors="pt").to(device)
with torch.no_grad():
    _ = model.get_image_features(**warmup_inputs)

total_start = time.time()
all_results = []

for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i + batch_size]
    batch_start = time.time()
    
    with torch.no_grad():
        img_inputs = processor(images=batch, return_tensors="pt").to(device)
        img_embeds = model.get_image_features(**img_inputs)
        img_embeds = torch.nn.functional.normalize(img_embeds, dim=1)
        
        similarities = img_embeds @ text_embeds.T
        probs = (similarities * model.logit_scale.exp()).softmax(dim=-1)
        
        batch_time = time.time() - batch_start
        per_image_ms = (batch_time / len(batch)) * 1000
        
        for j in range(len(batch)):
            best_idx = probs[j].argmax().item()
            score = probs[j][best_idx].item()
            all_results.append({
                "image_idx": valid_meta[i + j],
                "category": taxonomy_labels[best_idx],
                "confidence": f"{score:.4f}",
                "batch_time_ms": f"{per_image_ms:.1f}"
            })
    
    print(f"  Batch {i//batch_size + 1}: {len(batch)} images in {batch_time:.2f}s ({per_image_ms:.1f}ms/img)")

total_time = time.time() - total_start
print(f"\nTotal: {len(all_results)} images in {total_time:.2f}s")
print(f"Average: {(total_time / len(all_results)) * 1000:.1f}ms per image")
print(f"\nSample predictions:")
for r in all_results[:5]:
    print(f"  Image {r['image_idx']}: {r['category']} ({r['confidence']}) [{r['batch_time_ms']}ms/img]")
```

**Thresholding** determines what gets flagged for review. A softmax confidence of 0.35 across five categories means the model is uncertain — the top prediction barely edges out the others. Set a minimum confidence threshold (typically 0.4–0.6 for 5-category taxonomies) and route low-confidence predictions to manual review or a fallback enrichment step. This is the same score-and-qualify logic used elsewhere in the Clay waterfall: confidence below threshold means the record needs another enrichment source or human review.

The enrichment waterfall pattern — Find → Enrich → Transform → Export — treats CLIP as one stage in a multi-step pipeline. If CLIP's confidence is low, the waterfall continues to the next enrichment source (e.g., text-based classification of the HTML, or a third-party API). CLIP is not the final arbiter; it is one signal in a waterfall of signals, and its contribution is weighted by confidence. [CITATION NEEDED — concept: enrichment waterfall with confidence-based fallback]