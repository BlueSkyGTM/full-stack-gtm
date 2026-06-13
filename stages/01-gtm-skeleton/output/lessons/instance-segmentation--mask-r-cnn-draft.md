# Instance Segmentation — Mask R-CNN

## Hook It

Object detection tells you *where* bounding boxes are. Semantic segmentation tells you *which pixels belong to which class*. Neither tells you which pixels belong to **which individual object** when three people overlap in a frame. Instance segmentation solves this by producing a per-pixel mask for each detected instance, not each class.

## Define It

Mask R-CNN extends Faster R-CNN by adding a third output head: a binary segmentation mask per RoI. The mechanism is a three-branch parallel prediction — class label, bounding box regression, and pixel-level mask — run on each region proposal. The critical architectural detail is **RoIAlign**, which replaces RoIPool to eliminate the quantization misalignment that destroys mask precision. The mask branch outputs K masks (one per class) per RoI, but only the mask for the predicted class is selected at inference.

**Key terms to lock down:**
- Instance segmentation vs. semantic segmentation vs. panoptic segmentation
- RoIAlign (bilinear interpolation, no quantization)
- Mask branch architecture (FCN on each RoI)
- Loss function: $L = L_{cls} + L_{box} + L_{mask}$ where $L_{mask}$ is per-pixel binary cross-entropy averaged over the RoI

## Show It

Walk through a single forward pass on one image:

1. **Backbone + FPN** extracts multi-scale feature maps.
2. **RPN** proposes ~2000 candidate bounding boxes, filtered to ~1000 by NMS.
3. **RoIAlign** crops and resizes each proposal to a fixed 14×14 feature grid using bilinear interpolation (no rounding).
4. **Three parallel heads** on each RoI: classification (softmax over K classes), box regression (4 offsets), mask prediction (K binary masks via small FCN).
5. **Post-processing**: select mask for predicted class only, resize to original image dimensions, threshold at 0.5.

Show visual comparison: same street scene processed by (a) Faster R-CNN bounding boxes, (b) semantic segmentation class map, (c) Mask R-CNN instance masks with distinct colors per instance.

**Exercise hook (easy):** Given the three output heads and their losses, write out the total loss calculation for a single RoI predicted as "person" with IoU 0.82 and mask accuracy 0.91.

## Build It

Run inference with a pre-trained Mask R-CNN from torchvision on a real image:

```python
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
from torchvision import transforms

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

img = Image.open("street_scene.jpg")
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    predictions = model(img_tensor)

print(f"Detections: {len(predictions[0]['boxes'])}")
print(f"Classes: {predictions[0]['labels']}")
print(f"Scores: {predictions[0]['scores']}")
print(f"Mask shape: {predictions[0]['masks'].shape}")
print(f"First mask nonzero pixels: {(predictions[0]['masks'][0] > 0.5).sum().item()}")
```

This produces observable output: number of instances detected, their classes, confidence scores, mask tensor shape, and pixel counts per mask.

**Exercise hook (medium):** Modify the above to filter detections by score threshold (keep only >0.7), then count how many distinct person instances are present. Print the bounding box overlap (IoU) between each pair of person instances.

**Exercise hook (hard):** Implement RoIAlign manually on a 4×4 feature map for a single proposal with fractional coordinates. Apply bilinear interpolation, compare output to a quantized (RoIPool-style) crop, and print the pixel-wise difference.

## Use It

Instance segmentation maps to **Zone 1 (Signal Capture)** and **Zone 2 (Enrichment)** workflows where visual content must be parsed at the instance level.

**Specific GTM applications:**

- **Screenshot parsing for competitive intelligence:** Identify and extract individual UI components (modals, navbars, CTA buttons) from competitor screenshots. Instance segmentation distinguishes between three separate buttons on a page; bounding-box detection alone might merge them.
- **Document layout analysis:** Segment individual form fields, signature blocks, and table cells from inbound PDFs or scanned documents in an enrichment pipeline.
- **Logo detection and co-occurrence:** Detect multiple brand logos in a single image and track which instances overlap or co-occur — this feeds signal for sponsor/influencer identification.

If the workflow only needs "does this image contain X," use classification. If it needs "where is X," use detection. If it needs "which pixels belong to *each* X," use instance segmentation.

GTM redirect: **Clay enrichment workflows that ingest visual assets** — logo detection, screenshot analysis, document parsing. Instance segmentation is the mechanism when individual object boundaries matter, not just bounding boxes.

## Ship It

**Production constraints for Mask R-CNN:**

- **Inference speed:** ResNet-50-FPN backbone runs ~5 FPS on a single GPU. For batch processing (enrichment pipelines), this is acceptable. For real-time (video), swap to MobileNet backbone or export to ONNX/TensorRT.
- **Memory:** 4–6 GB VRAM per image at standard resolution. Batch multiple images only if memory permits.
- **Mask quality evaluation:** Use COCO metrics — AP@IoU:0.50:0.95 for mask mAP. Report both box AP and mask AP; if box AP is high but mask AP is low, the detection works but boundaries are imprecise.
- **Threshold tuning:** Default mask threshold is 0.5. For GTM use cases (logo extraction, document parsing), tune per class — higher threshold for precision-critical tasks, lower for recall.

**Exercise hook (hard):** Benchmark inference time on 50 images. Report mean FPS, median mask AP against ground truth (if available), and VRAM peak usage. Identify the bottleneck stage (RPN proposal generation vs. mask head inference) by timing each module separately.

---

**GTM Redirect Summary:** Instance segmentation is foundational for **Zone 1 Signal Capture** and **Zone 2 Enrichment** — specifically Clay workflows that parse screenshots, detect logos at the instance level, or extract structured regions from unstructured visual inputs. The redirect is: "this is the mechanism behind visual enrichment pipelines where individual object boundaries matter."