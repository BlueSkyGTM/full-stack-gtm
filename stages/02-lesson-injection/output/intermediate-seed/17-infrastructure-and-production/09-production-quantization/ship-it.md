## Ship It

Deploying a quantized model into production GTM infrastructure means versioning your quantized artifacts alongside your enrichment waterfalls and scoring logic. The deployment lifecycle mirrors MLOps: you version the base model, the quantization format, the calibration dataset, and the LoRA adapters as independent artifacts, then track how changes in each affect downstream GTM metrics — classification accuracy on real prospect data, enrichment completeness rates, and scoring calibration drift.

The two production traps are consistent across teams. First, the **calibration-dataset pitfall**: AWQ and GPTQ both require a calibration dataset to determine which weight channels are salient (AWQ) or to approximate the Hessian (GPTQ). If you calibrate on general text (Wikipedia, RedPajama) but deploy on domain-specific traffic (SaaS company descriptions, B2B intent signals, technical product documentation), the salient channels identified during calibration will not match the channels that matter at inference time. Your prospect-scoring agent will degrade silently — not catastrophically, but enough to shift your enrichment waterfall's precision by 3–8 percentage points. The fix is to calibrate on a sample of your actual deployment traffic: pull 1,000–2,000 representative prompts from your production logs and use those as the calibration set.

Second, the **KV cache blind spot**: teams quantize weights to INT4, see the model is now 4 GB, and deploy on a 16 GB GPU — then hit OOM at production batch sizes because the KV cache is 10–20 GB. KV cache quantization (KV cache INT8 or FP8) is separate from weight quantization and must be configured independently in vLLM or TensorRT-LLM. Your deployment checklist must account for both.

```python
deployment_checklist = {
    "Llama-3 8B AWQ INT4 on vLLM (L4 24GB)": {
        "weight_memory_gb": 4.0,
        "kv_cache_memory_gb": 8.0,
        "activation_buffers_gb": 2.0,
        "total_expected_gb": 14.0,
        "gpu_vram_gb": 24.0,
        "headroom_gb": 10.0,
        "max_batch_size_recommended": 64,
        "calibration_source": "production_prospect_logs_sample_2k.jsonl",
        "kv_cache_quantization": "FP8",
        "serving_engine": "vLLM 0.6.x with Marlin-AWQ kernels",
        "monitoring_metrics": [
            "tokens_per_second",
            "p99_latency_ms",
            "classification_accuracy_drift",
            "kv_cache_utilization_pct",
            "gpu_memory_fragmentation_ratio",
        ],
    },
    "Llama-3 8B GGUF Q4_K_M on Ollama (M2 Pro 16GB)": {
        "weight_memory_gb": 4.9,
        "kv_cache_memory_gb": 3.0,
        "total_expected_gb": 7.9,
        "host_ram_gb": 16.0,
        "headroom_gb": 8.1,
        "max_batch_size_recommended": 4,
        "calibration_source": "N/A (GGUF pre-quantized by model publisher)",
        "kv_cache_quantization": "FP16 (CPU inference, no quantization)",
        "serving_engine": "Ollama 0.3.x with llama.cpp backend",
        "monitoring_metrics": [
            "tokens_per_second",
            "memory_pressure",
            "thermal_throttling_events",
        ],
    },
}

for deployment, config in deployment_checklist.items():
    print(f"=== {deployment} ===\n")
    for key, value in config.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    print()

import hashlib

def compute_deployment_fingerprint(model_name, quant_format, calibration_hash, engine_version):
    fingerprint_input = f"{model_name}|{quant_format}|{calibration_hash}|{engine_version}"
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]

calibration_data_hash = hashlib.sha256(b"production_prospect_logs_sample_2k.jsonl").hexdigest()[:16]
fingerprint = compute_deployment_fingerprint(
    "Llama-3-8B-Instruct",
    "AWQ-INT4-g128",
    calibration_data_hash,
    "vLLM-0.6.3"
)

print(f"Deployment fingerprint: {fingerprint}")
print(f"  Tag your Docker image and model registry with this hash.")
print(f"  Any change to model, format, calibration data, or engine")
print(f"  produces a different hash — enabling rollback to exact")
print(f"  configurations when scoring drift is detected.")
```

The deployment fingerprint is the connective tissue between your quantization choice and your GTM lifecycle. When your enrichment waterfall's classification accuracy drifts — detected via scoring-model monitoring in your GTM dashboard — you can trace the drift back to a specific deployment fingerprint, identify whether the cause was a format change, a calibration dataset change, or an engine upgrade, and roll back to a known-good configuration. This is MLOps applied to GTM infrastructure: versioning your enrichment waterfalls, detecting when your scoring model drifts, and maintaining a living registry of deployed model configurations.