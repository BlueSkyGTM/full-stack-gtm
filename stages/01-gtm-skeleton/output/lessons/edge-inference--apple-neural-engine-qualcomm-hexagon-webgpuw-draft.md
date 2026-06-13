# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

## Beat 1: Hook

Edge inference is running transformer inference on silicon you physically control — no API call, no egress cost, no latency from a round-trip to us-east-1. Apple's Neural Engine, Qualcomm's Hexagon DSP, NVIDIA's Jetson GPU, and WebGPU in the browser are four deployment surfaces that each require different model formats, quantization schemes, and memory management strategies. The mechanism differs for each; the constraint is the same: limited memory bandwidth, thermals, and compute budgets that make a H100 look infinite.

## Beat 2: Concept

Description: Define the hardware accelerator classes — dedicated NPU (Apple ANE), DSP with vector extensions (Hexagon), integrated GPU with unified memory (Jetson), and browser-exposed GPU compute (WebGPU). Map each to its memory hierarchy, supported data types, and the compilation targets each requires (Core ML `.mlmodelc`, QNN `.so`, TensorRT `.engine`, WebGPU WGSL shaders). Explain why "runs on GPU" is ambiguous — a Jetson GPU with unified memory has different constraints than a discrete GPU with PCIe transfer overhead.

## Beat 3: Mechanism

Description: Walk through the inference compilation pipeline for each target. Apple: Core ML Tools converts PyTorch → TorchScript → Core ML with ANE scheduling. Qualcomm: AI Engine Direct (QNN) compiles to Hexagon HVX/HTA instructions with 8-bit quantization. Jetson: TensorRT builds a runtime engine with layer fusion and calibrates INT8. WebGPU/WebLLM: llama.cpp compiles to WGSL compute shaders with kv-cache in GPU buffers. The shared pattern: graph optimization → quantization → target-specific compilation → runtime binding. The divergence: memory management is unified on Jetson/Apple, copy-heavy on Hexagon, and sandboxed on WebGPU.

## Beat 4: Code

Description: Four runnable code blocks, each exercising one target. (1) `coremltools` Python script that converts a small PyTorch model to `.mlmodelc` and prints the compute unit assignment — observable output confirms ANE vs GPU vs CPU scheduling. (2) QNN SDK command that compiles an ONNX model to Hexagon and logs the operator partitioning. (3) TensorRT Python sample on Jetson that builds an engine, prints binding shapes, and reports throughput in tokens/second. (4) Browser-based WebLLM TypeScript snippet that loads a quantized model and prints token generation latency — runnable in Node via `@aspect-build/rules_js` or printed as a standalone script with instructions. Each script produces observable output: compute unit selection, operator mapping, throughput numbers, or latency measurements.

## Beat 5: Use It

Description: GTM redirect — edge inference maps to **Zone 1 (Prospecting Automation)** and **Zone 2 (Enrichment)** where field sales runs locally without network dependency. Specific application: deploy a small classification model on Jetson to score lead lists from a CSV at a trade show with no internet. The mechanism: ONNX → TensorRT INT8 engine processes rows at 500+ inferences/second on Jetson Orin Nano, writing scored output to local disk. This is the "offline scoring" pattern — same model, same weights, zero API cost, zero latency from cloud round-trips. [CITATION NEEDED — concept: offline edge inference for field sales lead scoring at events]

Exercise hooks:
- Easy: Modify the TensorRT script to change the batch size and observe throughput delta
- Medium: Quantize a model to INT8 with calibration data and compare accuracy drop vs FP16 on Jetson
- Hard: Compile the same PyTorch model to both Core ML and TensorRT; benchmark and explain the throughput difference in terms of memory bandwidth

## Beat 6: Ship It

Description: Production checklist for edge inference deployment. Model signing and versioning — Core ML requires `.mlmodelc` in an Apple-signed framework; TensorRT engines are device-specific and not portable across Jetson generations. Thermal throttling detection — log GPU temperature and clock speed during sustained inference. Memory pressure — Jetson's unified memory means inference competes with OS; set `GPU_DL_DEFDLY` and reserve memory. WebGPU production constraints: shader compilation is async and can fail on older browsers; implement fallback to WASM. Monitoring: emit inference latency percentiles (p50, p95, p99) and token throughput to local logs; sync when connectivity resumes. GTM redirect: this ships into Zone 1 as the "offline prospecting agent" — a self-contained edge device that scores, ranks, and prioritizes leads without cloud dependency.

Exercise hooks:
- Easy: Add thermal logging to the Jetson TensorRT script; observe clock throttling under sustained load
- Medium: Implement a fallback chain: attempt WebGPU, catch compilation failure, fall back to WASM via `onnxruntime-web`, print which backend activated
- Hard: Build a release script that cross-compiles a single PyTorch model to Core ML, QNN, and TensorRT targets; validate each with the same test inputs and assert output parity within quantization tolerance