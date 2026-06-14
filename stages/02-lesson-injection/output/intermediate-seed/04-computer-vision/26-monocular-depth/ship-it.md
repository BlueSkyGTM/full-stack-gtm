## Ship It

Putting depth estimation into production means handling the failure modes and providing scale references. Here is a complete pipeline that runs depth estimation, computes confidence via local variance, flags potential failures, and applies a scale reference to convert relative depth to approximate metric depth:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from scipy.ndimage import uniform_filter

pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

image_size = 480
image_array = np.full((image_size, image_size, 3), [80, 80, 100], dtype=np.uint8)

y_coords, x_coords = np.ogrid[:image_size, :image_size]

floor_mask = y_coords > 300
image_array[floor_mask] = [100, 90, 70]

wall_mask = y_coords <= 300
image_array[wall_mask] = [140, 140, 150]

rect_mask = (x_coords > 180) & (x_coords < 280) & (y_coords > 200) & (y_coords < 350)
image_array[rect_mask] = [180, 100, 80]

thin_mask = (x_coords > 350) & (x_coords < 355) & (y_coords > 150) & (y_coords < 400)
image_array[thin_mask] = [200, 200, 50]

noise = np.random.randint(-10, 10, (image_size, image_size, 3), dtype=np.int16)
image_array = np.clip(image_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

image = Image.fromarray(image_array).convert("RGB")

result = pipe(image)
depth = result["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float64)

local_mean = uniform_filter(depth, size=15)
local_sqr_mean = uniform_filter(depth**2, size=15)
local_var = local_sqr_mean - local_mean**2
local_std = np.sqrt(np.maximum(local_var, 0))

global_std = depth.std()
confidence = 1.0 - np.clip(local_std / (global_std + 1e-8), 0, 1)

low_conf_mask = confidence < 0.3
low_conf_pct = (low_conf_mask.sum() / low_conf_mask.size) * 100

print("=== Production Depth Pipeline ===\n")
print(f"Depth map shape: {depth.shape}")
print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"Global std: {global_std:.4f}")
print(f"Low-confidence pixels (<0.3): {low_conf_mask.sum()} ({low_conf_pct:.1f}% of image)")
print(f"Mean confidence: {confidence.mean():.4f}")

thin_region_depth = depth[150:400, 350:355]
surrounding_depth = depth[150:400, 340:365]
print(f"\nThin structure region std: {thin_region_depth.std():.4f}")
print(f"Surrounding region std:    {surrounding_depth.std():.4f}")
print(f"Thin structure depth anomaly: {'YES - high variance near thin object' if thin_region_depth.std() > surrounding_depth.std() * 1.5 else 'NO'}")

print("\n=== Scale Reference Application ===")
known_object_pixels = 100
known_object_meters = 0.5
scale_factor = known_object_meters / known_object_pixels

metric_depth = depth * scale_factor
print(f"Known reference: {known_object_pixels} pixels = {known_object_meters} meters")
print(f"Scale factor: {scale_factor:.6f} meters/pixel-depth-unit")
print(f"Metric depth range: [{metric_depth.min():.4f}, {metric_depth.max():.4f}] meters")
print(f"Relative depth was: [{depth.min():.4f}, {depth.max():.4f}]")
print(f"Note: scale is approximate and assumes linear depth scale (valid near camera, degrades at distance)")

print("\n=== Enrichment Waterfall: Confidence Tiers ===")
tier1 = confidence[confidence >= 0.7]
tier2 = confidence[(confidence >= 0.4) & (confidence < 0.7)]
tier3 = confidence[confidence < 0.4]
print(f"Tier 1 (high confidence, >=0.7):   {len(tier1)} pixels ({len(tier1)/confidence.size*100:.1f}%) -> verified attribute")
print(f"Tier 2 (medium confidence, 0.4-0.7): {len(tier2)} pixels ({len(tier2)/confidence.size*100:.1f}%) -> inferred attribute")
print(f"Tier 3 (low confidence, <0.4):     {len(tier3)} pixels ({len(tier3)/confidence.size*100:.1f}%) -> missing/handoff")

print("\nWaterfall routing decision:")
print(f"  Route Tier 1 directly to downstream (3D lift, navigation, export)")
print(f"  Route Tier 2 to verification pass (second model, consistency check)")
print(f"  Route Tier 3 to fallback (ignore, use prior, or flag for review)")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Input Image")
axes[0, 0].axis("off")

depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
axes[0, 1].imshow(depth_vis, cmap="inferno")
axes[0, 1].set_title("Relative Depth Map")
axes[0, 1].axis("off")

im_conf = axes[1, 0].imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
axes[1, 0].set_title("Per-Pixel Confidence")
axes[1, 0].axis("off")
plt.colorbar(im_conf, ax=axes[1, 0], label="Confidence")

metric_vis = (metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min() + 1e-8)
axes[1, 1].imshow(metric_vis, cmap="viridis")
axes[1, 1].set_title("Approximate Metric Depth (scaled)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("production_depth.png", dpi=150)
print("\nSaved to production_depth.png")
```

The confidence map uses local variance as a proxy for reliability. High local variance in depth means the model is producing inconsistent predictions across neighboring pixels — likely a thin structure, an edge artifact, or a transparent surface. Low local variance means the depth is smooth and internally consistent, which correlates with (but does not prove) correctness. This is the same logic as scoring enrichment records: a record where three providers agree on the same email is high-confidence; a record where providers disagree is routed to a second pass.

In a production GTM pipeline, this confidence tiering maps directly to how Clay exports data. [CITATION NEEDED — concept: Clay's scoring and qualification routing thresholds] High-confidence depth pixels are like enriched records where behavioral signals (feature usage depth, session frequency, team size) align across sources — you route them directly to activation. Medium-confidence pixels get a second model pass, just as ambiguous enrichment records get a manual review or a verification email. Low-confidence pixels are flagged and ignored, like enrichment records that return no data from any provider in the waterfall.

The scale reference application at the end is the most important production decision. If your downstream system needs metric depth (robotics, AR with physical placement, autonomous navigation), you must provide a known reference. In enrichment terms, this is the difference between knowing "this company seems engaged" (relative depth) and knowing "this company has 15 daily active users on our platform" (metric depth). The behavioral signals — feature usage depth, session frequency — are the known object width that converts relative intent into measured engagement. [CITATION NEEDED — concept: linking product behavioral signals to enrichment scoring in Clay]