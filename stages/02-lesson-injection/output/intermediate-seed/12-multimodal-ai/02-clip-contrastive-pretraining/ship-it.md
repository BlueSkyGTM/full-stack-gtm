## Ship It

Deploying CLIP-based visual enrichment into a production pipeline means wrapping the inference call in observability. The Zone 12 pattern — pipeline health monitoring through signal drift — applies directly: when average cosine similarity scores on your page-type classifier drop below a threshold, your enrichment quality is degrading, and you need to know before downstream sequences start misfiring.

The mechanism for monitoring: log every classification's top score and confidence margin (difference between the top two scores). Track the distribution over time. A healthy pipeline shows stable score distributions. A drift in those distributions indicates either model degradation (unlikely with a frozen CLIP checkpoint) or input distribution shift (your prospect list now contains pages CLIP was never trained on — foreign-language sites, heavy JavaScript SPAs that render as blank screenshots, or new design patterns). This is the visual analog of reply rate drift as a model degradation signal: the score distribution is your leading indicator, and you catch the problem before it corrupts your CRM data.

```python
import json
import time
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

PAGE_TYPES = ["pricing page", "blog post", "login page", "product page", "contact page"]
PROMPTS = [f"a screenshot of a {pt}" for pt in PAGE_TYPES]

text_inputs = processor(text=PROMPTS, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    text_outputs = model.get_text_features(**text_inputs)
    prompt_embeds = F.normalize(text_outputs, dim=-1)

def classify_screenshot(image, source_url="", min_confidence=0.15):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        img_embeds = F.normalize(model.get_image_features(**inputs), dim=-1)

    scores = (img_embeds @ prompt_embeds.T).squeeze(0)

    top_idx = scores.argmax().item()
    top_score = scores[top_idx].item()
    sorted_scores = scores.sort(descending=True)
    margin = (sorted_scores.values[0] - sorted_scores.values[1]).item()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": source_url,
        "predicted_type": PAGE_TYPES[top_idx],
        "top_score": round(top_score, 4),
        "confidence_margin": round(margin, 4),
        "all_scores": {PAGE_TYPES[i]: round(scores[i].item(), 4) for i in range(len(PAGE_TYPES))},
        "low_confidence_flag": top_score < min_confidence,
    }
    return result

mock_pages = [
    ("acme.com/pricing", Image.new("RGB", (400, 300), (240, 240, 240))),
    ("acme.com/blog/launch", Image.new("RGB", (400, 300), (250, 250, 245))),
    ("acme.com/login", Image.new("RGB", (400, 300), (245, 245, 248))),
]

print("Production CLIP Enrichment Pipeline — Mock Run")
print("=" * 65)
print(f"Prompt bank: {len(PROMPTS)} page types")
print(f"Min confidence threshold: 0.15")
print()

score_log = []
for url, img in mock_pages:
    result = classify_screenshot(img, source_url=url)
    score_log.append(result["top_score"])

    flag = " *** LOW CONFIDENCE — REVIEW" if result["low_confidence_flag"] else ""
    print(f"  URL: {url}")
    print(f"  Type: {result['predicted_type']}")
    print(f"  Score: {result['top_score']:.4f}  Margin: {result['confidence_margin']:.4f}{flag}")
    print(f"  All scores: {json.dumps(result['all_scores'], indent=4)}")
    print()

avg_score = sum(score_log) / len(score_log)
min_seen = min(score_log)
print(f"Batch Summary")
print(f"  Mean top score:    {avg_score:.4f}")
print(f"  Min top score:     {min_seen:.4f}")
print(f"  Low confidence ct: {sum(1 for s in score_log if s < 0.15)}/{len(score_log)}")
print()
print("In production: emit score_log to your observability backend.")
print("Alert when rolling mean drops >2σ from baseline.")
print("Route low_confidence_flag items to human review queue.")
```

The `min_confidence` threshold and the rolling mean alert are the two levers that connect CLIP inference to GTM pipeline health. Every enrichment record either lands in the CRM as a confident classification or gets flagged for review. The score log becomes a time series you monitor exactly like email reply rates or meeting booking rates — when it drifts, something upstream changed, and you investigate before the bad data propagates into sequencing decisions.

The blank-image mock will produce low confidence scores across all page types. That is the correct behavior — CLIP has no strong prior about page type when the input carries no discriminative visual content. In a real pipeline, these low-confidence items are exactly the screenshots you want to catch and route to a human reviewer or flag as "screenshot capture failed" before they generate a misleading enrichment record.