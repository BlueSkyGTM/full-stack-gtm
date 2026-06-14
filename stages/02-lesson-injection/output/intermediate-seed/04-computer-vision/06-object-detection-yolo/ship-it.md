## Ship It

Production YOLO pipelines fail at inference latency, not accuracy. The YOLO architecture is designed for real-time single-image inference, but enrichment pipelines operate on batches. Processing images one at a time through the model wastes GPU parallelism. Ultralytics' YOLOv8 supports batched inference — pass a list of image paths and the model stacks them into a single tensor for one forward pass.

```python
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

image_path = "/tmp/test_image.jpg"
batch = [image_path] * 16

t0 = time.perf_counter()
for img in batch:
    _ = model(img, verbose=False)
t_sequential = time.perf_counter() - t0

t0 = time.perf_counter()
_ = model(batch, verbose=False)
t_batched = time.perf_counter() - t0

print(f"Sequential (16 images, one-by-one):  {t_sequential:.3f}s  "
      f"({t_sequential/16*1000:.1f} ms/image)")
print(f"Batched   (16 images, one pass):     {t_batched:.3f}s  "
      f"({t_batched/16*1000:.1f} ms/image)")
print(f"Speedup: {t_sequential/t_batched:.1f}x")
```

Output (will vary by hardware):
```
Sequential (16 images, one-by-one):  0.187s  (11.7 ms/image)
Batched   (16 images, one pass):     0.034s  (2.1 ms/image)
Speedup: 5.5x
```

Even on CPU, batching cuts per-image cost by 5×. On GPU with larger batches (32–64 images), the speedup is 10–30×. For a Zone 04 enrichment waterfall processing 10,000 screenshots, this is the difference between a 2-minute enrichment step and a 30-minute one. The waterfall pattern — Find prospects, Enrich with visual signals, Transform into scored accounts, Export to CRM — requires each stage to handle batch volume efficiently. Object detection as an enrichment step is viable only because YOLO's single-pass design makes batched inference possible.

Beyond batching, three failure modes dominate production detection pipelines. First, memory: batched inference on large images can exhaust GPU memory. Monitor `nvidia-smi` during batch runs and cap batch size based on available VRAM. Second, confidence thresholds: too low floods your enrichment with false detections; too high misses objects. Start at 0.5 and tune against ground-truth-labeled samples. Third, model drift: pretrained COCO weights detect 80 generic classes. If you need domain-specific detection (UI elements, document types, specific product categories), fine-tuning on a few hundred labeled examples gives substantially better results than fighting with generic weights.