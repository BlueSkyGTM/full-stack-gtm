# Object Detection — YOLO from Scratch

## Hook
Object detection requires both *what* and *where*. Sliding-window classifiers run hundreds of passes per image. YOLO reframes detection as a single regression pass over a grid — one forward pass, all predictions at once. If you're running visual analysis on prospect websites at scale, per-image latency determines whether you process 100 accounts or 10,000.

## Learn It
YOLO divides an input image into an S×S grid. Each grid cell predicts B bounding boxes (x, y, width, height) plus an objectness confidence score and C class probabilities. The output tensor is S × S × (B × 5 + C). A confidence threshold filters low-certainty boxes. Non-maximum suppression (NMS) collapses overlapping detections into single predictions using Intersection-over-Union (IoU) as the merge criterion. The multi-part loss function penalizes coordinate error, confidence error, and classification error simultaneously — trained end-to-end on annotated datasets (e.g., COCO, Pascal VOC).

## See It
Implement IoU computation and NMS from scratch on synthetic box coordinates. Print IoU values between overlapping and non-overlapping rectangles. Run NMS on a set of 8 overlapping boxes with varying confidence scores and print the surviving boxes. Then load a pretrained YOLOv8-nano model (Ultralytics), run inference on a test image from COCO, and print detected class labels, confidence scores, and bounding box coordinates to terminal.

## Use It
**GTM Redirect — Zone 01 (Prospect Intelligence):** Visual signal extraction from company websites. Object detection identifies logos, team photos, product screenshots, and document types on prospect pages. These signals feed account scoring models — a company with 50+ product screenshots on their site has different intent patterns than one with none. Build a function that takes a directory of scraped homepage screenshots, runs YOLO detection, counts detected object categories per account, and outputs a CSV of visual signal density per company. [CITATION NEEDED — concept: visual signal density as account scoring input in Zone 01]

## Build It
**Easy:** Implement IoU between two axis-aligned bounding boxes. Test on known overlapping and non-overlapping pairs, print results.  
**Medium:** Implement NMS from scratch given a list of (x1, y1, x2, y2, confidence) detections and an IoU threshold. Feed it 10 synthetic boxes with deliberate overlaps and print the suppressed vs. surviving boxes.  
**Hard:** Build a minimal single-class YOLO-like detector for MNIST digits placed randomly on a blank canvas. Generate a dataset of images with 1–3 digits at random positions. Implement a small CNN that outputs a 7×7 grid with (objectness, x, y, w, h, class) per cell. Train for 10 epochs and print detection results on 5 test images.

## Ship It
Production YOLO pipelines fail at inference latency, not accuracy. Batch inference on GPU processes 32+ images per forward pass — sequential single-image calls are 10–30× slower. Model size matters at edge: YOLOv8n is 6MB, YOLOv8x is 225MB. Export to ONNX or TensorRT for consistent cross-platform inference without PyTorch runtime overhead. For GTM-scale web screenshot processing, a batched YOLOv8n-ONNX pipeline processes ~200 images/second on a single T4 GPU — enough to analyze 1M accounts in ~80 minutes. Log per-image inference time and detection counts to detect drift (new website formats producing fewer detections may indicate template changes, not model degradation).