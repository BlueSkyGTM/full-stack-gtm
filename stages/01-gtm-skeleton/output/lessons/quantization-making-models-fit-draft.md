# Quantization: Making Models Fit

## Beat 1: Hook — Why Your Model Doesn't Fit

A 7B parameter model at FP32 needs ~28GB of VRAM just to load. Most inference GPUs have 24GB or less. Quantization is the compression technique that closes this gap by reducing the numerical precision of weights and/or activations. Without it, you're renting A100s for tasks that could run on a laptop.

## Beat 2: Mechanism — How Precision Reduction Works

Explain the core algorithm: mapping continuous FP32/FP16 values to discrete lower-bit integers (INT8, INT4, INT3). Cover the two dominant approaches — post-training quantization (PTQ) which recalibrates a trained model, and quantization-aware training (QAT) which simulates quantization during training. Explain symmetric vs. asymmetric quantization, the role of calibration datasets, and why outliers in activations make INT4 harder than it looks. Name the de facto standards: GPTQ (layer-wise Hessian approximation), GGUF/llama.cpp (runtime-quantized inference for CPU/consumer hardware), and bitsandbytes (on-the-fly quantization for training and inference). Mechanism first: the Hessian-based weight grouping that GPTQ uses to minimize reconstruction error, then the tool name.

## Beat 3: Code — Quantize and Measure

Load a small model (e.g., `twitter/t5-base` or similar), quantize it using bitsandbytes (NF4 config), and print: original size, quantized size, and a sample inference output to prove it still works. All in a single runnable Python script with `torch`, `transformers`, and `bitsandbytes`. Observable output: memory footprint before and after, plus generated text.

## Beat 4: Use It — GTM Redirect

Quantization directly enables cost-controlled inference in GTM tooling. Specific redirect: running quantized local models for lead scoring, enrichment, or classification inside Clay workflows where API per-token costs would make large-volume processing uneconomical. If you're classifying 100K companies with a custom model, FP16 on cloud GPUs might cost $X; INT4 on a single GPU costs $X/10. The Clay waterfall pattern — enrich → classify → route — benefits from quantized models when the classification step uses a custom LLM rather than a remote API. [CITATION NEEDED — concept: Clay waterfall local model integration for classification steps]

## Beat 5: Exercises

**Easy:** Write a script that loads any HuggingFace model, prints its size in FP32, then reloads it in 8-bit using bitsandbytes and prints the new size. Confirm the ratio.

**Medium:** Quantize the same model at both INT8 and INT4 (NF4), run the same five prompts through each, and compute the perplexity difference (or exact match on structured outputs). Print a comparison table.

**Hard:** Implement symmetric linear quantization from scratch (no bitsandbytes): take an FP32 weight tensor, compute scale and zero-point, quantize to INT8, dequantize back, and print the MSE between original and reconstructed weights. Compare your MSE to bitsandbytes output on the same layer.

## Beat 6: Ship It — Production Checkpoint

Build a quantized inference endpoint that loads a model in INT4, serves it via a minimal FastAPI `/generate` endpoint, and logs VRAM usage at startup. Test with a curl request. The deliverable is a deployable script that runs on a T4 (16GB) and proves the model fits when it wouldn't at FP16. GTM redirect: this is the infrastructure pattern for running custom classification models behind Clay webhooks at scale, where you control inference cost per enrichment call.