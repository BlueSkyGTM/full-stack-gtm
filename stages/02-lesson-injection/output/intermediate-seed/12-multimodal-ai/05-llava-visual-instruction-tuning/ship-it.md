## Ship It

Multimodal inference concatenates 576 visual tokens with text tokens before the transformer's forward pass, so peak VRAM scales with both image resolution and prompt length. For the 7B variant at fp16, the model weights alone consume ~13 GB. At 4-bit quantization via bitsandbytes, that drops to ~4 GB of weights, with the remaining VRAM budget consumed by the KV cache for the concatenated image-plus-prompt sequence. A 7B model in 4-bit with a 600-token visual prefix and 200-token generation fits comfortably in 8 GB VRAM.

The serving pattern that matters for production is image caching. If your enrichment pipeline runs multiple prompts against the same screenshot—first extracting product info, then pricing, then tech stack—re-encoding the image through CLIP on each call wastes compute. Cache the projected visual embeddings after the first ViT forward pass and reuse them. This is the multimodal equivalent of caching embeddings in a text retrieval pipeline.

```python
import torch
import gc

def vram_report(label=""):
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available — CPU mode")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {peak:.2f} GB")

vram_report("before model load")

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    model_id = "llava-hf/llava-1.5-7b-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    vram_report("after model load")
    
    model_vocab_size = model.config.text_config.vocab_size
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Image token index: {model.config.image_token_index}")
    
except Exception as e:
    print(f"Model load skipped (no GPU or weights): {e}")
    print("In production: load with BitsAndBytesConfig(load_in_4bit=True)")
    print("Expected VRAM at 4-bit: ~6 GB for 7B variant")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
vram_report("after cleanup")
```

This runs on any machine—it detects CUDA availability and reports actual VRAM if present, or prints the expected values if not. The `torch.cuda.max_memory_allocated()` call gives you the peak, which is the number that matters for capacity planning. In a GTM enrichment service processing screenshots at throughput, peak VRAM during the forward pass—not steady-state—is what causes OOM crashes under batch load.

For observability, the VRAM metrics from this function feed directly into the tracing layer described in the GTM engineering handbook's Zone 12 framework. [CITATION NEEDED — concept: specific VRAM/alerting thresholds for multimodal enrichment pipelines in production GTM systems]. The operational signal is drift: if the model starts producing shorter outputs, truncating responses, or returning empty strings on screenshots that previously yielded clean extractions, that is your enrichment-quality degradation signal. Log output length distributions and extraction success rates per batch, not just latency and VRAM. A multimodal model that hallucinates pricing tiers from screenshots because the projection layer degraded during a bad fine-tune will still return 200 OK with plausible-looking text—only structured output validation catches that.