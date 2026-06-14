## Ship It

In a production pipeline, the bottleneck is not the diffusion model — it is mask generation. You cannot hand-draw masks for 10,000 product images. You need automated masking that is reliable enough to run without human review on at least 80% of inputs. The remaining 20% get routed to a human reviewer.

Color thresholding handles green-screen and solid-background product photography. Semantic segmentation handles the harder case: removing a specific object class (person, car, logo) from an arbitrary background. Models like Segment Anything (SAM) or YOLO-segmentation produce pixel-accurate masks from a point prompt or bounding box, which you then feed into the inpainting pipeline.

```python
import numpy as np
from PIL import Image, ImageDraw
import time

class InpaintingPipeline:
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting",
                 min_mask_area=0.01, max_mask_area=0.75):
        self.min_mask_area = min_mask_area
        self.max_mask_area = max_mask_area
        self.pipe = None
        self.stats = {"processed": 0, "skipped_small": 0,
                      "skipped_large": 0, "errors": 0}

    def _load_model(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForInpainting
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_id if hasattr(self, 'model_id') else "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def validate_mask(self, mask_array):
        coverage = mask_array.mean() / 255.0
        if coverage < self.min_mask_area:
            return False, f"mask too small ({coverage*100:.1f}% < {self.min_mask_area*100:.1f}%)"
        if coverage > self.max_mask_area:
            return False, f"mask too large ({coverage*100:.1f}% > {self.max_mask_area*100:.1f}%)"
        return True, f"coverage {coverage*100:.1f}%"

    def batch_process(self, items, prompt_fn=None):
        default_prompt = "clean background, seamless, professional"
        results = []
        for item in items:
            image = item["image"]
            mask = item["mask"]
            prompt = prompt_fn(item) if prompt_fn else default_prompt

            mask_arr = np.array(mask)
            valid, reason = self.validate_mask(mask_arr)

            if not valid:
                if "too small" in reason:
                    self.stats["skipped_small"] += 1
                else:
                    self.stats["skipped_large"] += 1
                results.append({"id": item.get("id"), "status": "skipped",
                                "reason": reason})
                continue

            results.append({"id": item.get("id"), "status": "queued",
                            "prompt": prompt, "coverage": reason})
            self.stats["processed"] += 1

        return results

source1 = Image.new("RGB", (512, 512), (200, 200, 200))
draw1 = ImageDraw.Draw(source1)
draw1.rectangle([200, 200, 250, 250], fill=(255, 0, 0))

source2 = Image.new("RGB", (512, 512), (200, 200, 200))
mask2 = Image.new("L", (512, 512), 255)

source3 = Image.new("RGB", (512, 512), (200, 200, 200))
mask3 = Image.new("L", (512, 512), 0)
draw3 = ImageDraw.Draw(mask3)
draw3.ellipse([210, 210, 230, 230], fill=255)

mask1 = Image.new("L", (512, 512), 0)
draw_m1 = ImageDraw.Draw(mask1)
draw_m1.rectangle([200, 200, 250, 250], fill=255)

batch = [
    {"id": "IMG-001", "image": source1, "mask": mask1,
     "defect": "red logo on wall"},
    {"id": "IMG-002", "image": source2, "mask": mask2,
     "defect": "full frame overwrite"},
    {"id": "IMG-003", "image": source3, "mask": mask3,
     "defect": "tiny dust spot"},
]

def prompt_for_item(item):
    defect = item.get("defect", "object")
    return f"remove {defect}, clean surface, seamless background"

pipeline = InpaintingPipeline(min_mask_area=0.005, max_mask_area=0.60)
results = pipeline.batch_process(batch, prompt_fn=prompt_for_item)

print(f"Batch size: {len(batch)}")
print(f"Processed (queued): {pipeline.stats['processed']}")
print(f"Skipped (mask too small): {pipeline.stats['skipped_small']}")
print(f"Skipped (mask too large): {pipeline.stats['skipped_large']}")
print()
for r in results:
    print(f"  {r['id']}: {r['status']} — {r.get('reason', r.get('prompt', ''))}")
```

The mask validation thresholds — 1% minimum, 75% maximum — are not arbitrary. A mask covering less than 1% of the image is likely a detection false positive (a speck of noise flagged as an object). A mask covering more than 75% means the "edit" is essentially a full regeneration, and you should be using img2img instead, which is cheaper and does not require the 9-channel inpainting model. These thresholds should be tuned on your specific dataset: product photography on white backgrounds tends to have small masks (remove a small blemish), while scene editing has larger masks (replace entire backgrounds).

The batch processor pattern above mirrors what a Clay waterfall does in the CRM enrichment context. A Clay waterfall tries multiple data providers in sequence for each enriched field, stopping when it gets a hit. The inpainting pipeline tries a single model but applies the same selective-attention logic: skip records that do not need processing, flag records that are out of distribution, and process the rest. In both cases, the engineering challenge is the same — handling edge cases at scale without human intervention.