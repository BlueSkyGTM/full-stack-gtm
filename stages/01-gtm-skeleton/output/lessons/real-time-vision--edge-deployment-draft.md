# Real-Time Vision — Edge Deployment

---

## Beat 1: Hook

Frame the latency wall. Cloud inference at 30fps costs real money and adds 100–300ms round-trip per frame — unacceptable for any system that must react within a single camera frame. Edge deployment collapses that round-trip to local inference. The practitioner sees why "just call an API" stops working past a frame-rate threshold.

---

## Beat 2: Concept

Explain the inference pipeline reduction: from raw frames → preprocess → model forward pass → postprocess, and how each stage is a latency budget target. Cover the three canonical optimization mechanisms (quantization, pruning, operator fusion) before naming any framework. Then map each mechanism to its tool: TensorRT (operator fusion + INT8 quantization), ONNX Runtime (cross-platform graph execution), OpenVINO (Intel silicon-specific optimization). Finish with the hardware constraint equation: memory bandwidth × compute throughput ≥ frames per second × operations per frame.

---

## Beat 3: Code

Provide two runnable scripts. First: export a torchvision image classification model to ONNX, load it with ONNX Runtime, benchmark inference latency over 100 iterations, and print mean/std to terminal. Second: apply dynamic quantization to the same model, re-export, re-benchmark, print the delta. Both scripts run in terminal with no display dependency. Observable output is timing data confirming quantization speedup.

---

## Beat 4: Use It

GTM cluster: Zone 1 — Enrichment. Real-time edge vision classifies objects, scenes, or document types from camera feeds or video attachments. That classification output feeds enrichment pipelines — tagging trade show booth visitors by badge type, sorting uploaded document photos into routing buckets, or detecting product presence on shelves for channel monitoring. The redirect: this is the edge inference layer that makes enrichment from visual signals feasible at scale without per-frame API costs. If the GTM connection feels thin for a specific cohort, fallback to "foundational for Zone 1 enrichment pipelines that ingest visual signals."

---

## Beat 5: Ship It

Deployment checklist: model serialization format selection (ONNX vs SavedModel vs TFLite), target hardware profiling, batch-size tuning, thermal throttling detection, and graceful degradation when inference misses frame deadlines. Provide a terminal-runnable script that simulates a frame-processing loop with configurable FPS target, logs dropped frames, and prints a summary throughput report. This is the production heartbeat.

---

## Beat 6: Evaluate

- **Easy hook:** Given a printed inference latency budget breakdown, identify which pipeline stage to optimize first and justify why.
- **Medium hook:** Run the provided ONNX export + benchmark scripts on a provided model. Report which quantization level (FP32 vs FP16 vs INT8) meets a stated latency threshold. Show the terminal output.
- **Hard hook:** Modify the frame-processing loop to implement dynamic frame skipping when inference latency exceeds the FPS budget threshold. Print a log showing adaptive behavior under simulated load. Explain the accuracy–latency tradeoff introduced.

---

**GTM Citation Status:** [CITATION NEEDED — concept: Zone 1 enrichment pipeline ingesting visual signals for real-time classification and routing. Specific Clay workflow or GTM play referencing image/video-based enrichment not yet mapped in gtm-topic-map.md.]