## Ship It

To put Mask R-CNN into production on a custom dataset, you replace the pre-trained heads (which output COCO's 80 classes) with fresh heads for your classes, then fine-tune. The backbone stays frozen (or lightly tuned) because the features it learned on COCO generalize well. This is the same pattern as swapping out the enrichment provider list in a Clay waterfall — the pipeline structure (Find → Enrich → Transform → Export) stays the same, but the specific tools at each stage change to match the target market.

Here is a complete fine-tuning setup for a small custom dataset. The code below creates a two-class detector (background + one foreground class), replaces the relevant heads, and runs a training loop:

```python
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model

num_classes = 2
model = build_model(num_classes)

for name, param in model.backbone.named_parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

dummy_target = {
    "boxes": torch.tensor([[50.0, 50.0, 200.0, 200.0]]),
    "labels": torch.tensor([1], dtype=torch.int64),
    "masks": torch.zeros(1, 300, 300, dtype=torch.uint8),
    "masks": torch.randint(0, 2, (1, 300, 300), dtype=torch.uint8),
    "image_id": torch.tensor([0]),
    "area": torch.tensor([22500.0]),
    "iscrowd": torch.tensor([0], dtype=torch.int64),
}

dummy_images = [torch.rand(3, 300, 300)]
model.train()
losses = model(dummy_images, [dummy_target])
total_loss = sum(loss for loss in losses.values())
print(f"\nLoss components:")
for name, loss in losses.items():
    print(f"  {name}: {loss.item():.4f}")
print(f"  total: {total_loss.item():.4f}")

total_loss.backward()
print("\nBackward pass completed. Gradients computed on head parameters.")
grad_count = sum(
    1 for p in model.parameters()
    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
)
print(f"Parameters with nonzero gradients: {grad_count}")
```

A few production notes. First, the mask loss is computed at 28×28 resolution internally even though the final masks are upscaled to image dimensions at inference. This means mask quality is fundamentally limited by that resolution — tiny objects get blurry masks because 28×28 doesn't have enough pixels to represent them. Second, the inference is expensive: COCO has 80 classes, so the mask branch computes 80 masks per RoI even though only one is used. For a 2-class fine-tune, set `num_classes` correctly and the mask branch only computes 2 masks per RoI, which is 40× cheaper. Third, NMS in Mask R-CNN operates on bounding boxes (not masks), so overlapping masks from different instances can still overlap at the pixel level. The masks are not mutually exclusive — if two instances overlap, both masks can claim the same pixel. Post-processing with something like a mask voting scheme or per-pixel argmax across instances can fix this if needed.

For deployment, the standard pattern is to export to ONNX or TorchScript, cap the number of detections at inference (the `model.roi_heads.detections_per_img` parameter, default 100), and set a score threshold (default 0.05 in torchvision — far too low for production; 0.5 or 0.7 is more typical). The enrichment waterfall analogy holds here too: in a Clay waterfall, you set filters at each stage to drop low-quality results before passing to the next stage, which is exactly what the score threshold and NMS do in Mask R-CNN — prune bad proposals early so downstream stages don't waste computation on them.