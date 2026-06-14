## Ship It

In production, you have two deployment options for a LoRA-adapted model: keep the adapter separate (load base + adapter at runtime) or merge the adapter weights into the base model (single checkpoint). Merging eliminates the runtime overhead of computing $BAx$ on every forward pass — the adapter math gets baked into $W$. This matters for inference latency, especially when you're classifying thousands of replies per day in a real-time enrichment pipeline.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import time

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./lora-reply-classifier"

print("Loading base model in fp16 (no quantization for merge)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Attaching LoRA adapter...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter weights into base model...")
merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained("./merged-reply-classifier")
tokenizer.save_pretrained("./merged-reply-classifier")
print("Merged model saved to ./merged-reply-classifier")

prompt = "### Instruction: Classify this reply into one of: interested, not_now, wrong_person, unsubscribe\n\n### Input: Let's talk pricing.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

n_iters = 20
with torch.no_grad():
    for _ in range(3):
        merged_model.generate(**inputs, max_new_tokens=5, do_sample=False)

    start = time.time()
    for _ in range(n_iters):
        merged_model.generate(**inputs, max_new_tokens=5, do_sample=False)
    merged_time = (time.time() - start) / n_iters

print(f"\nMerged model avg inference: {merged_time*1000:.1f} ms/call")

output = merged_model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(f"Output: {tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)}")
```

The merged model is a standard Hugging Face checkpoint — no `peft` dependency at inference time. You can serve it with vLLM, TGI, or ONNX Runtime. The trade-off: the merged checkpoint is full model size (~2.2GB for TinyLlama, ~16GB for Llama 3 8B), so you lose the storage advantage of the adapter-only file. The right choice depends on your deployment constraints. If you're serving one task from one model, merge. If you're serving multiple tasks (reply classification + email drafting + ICP scoring) from the same base model, keep adapters separate and swap them at runtime — this is the multi-tenant LoRA serving pattern that vLLM supports natively.

For a GTM stack, the deployment pattern looks like this: your enrichment tool (Clay, ZoomInfo, or a custom pipeline) sends inbound replies to your fine-tuned model endpoint via API. The model returns a label (`interested`, `not_now`, `wrong_person`, `unsubscribe`). Your routing logic acts on that label: interested → trigger SDR sequence in your sales engagement platform; wrong_person → update the contact in Clay and re-enrich; unsubscribe → suppress and log for compliance. [CITATION NEEDED — concept: model-served reply classification integrated into Clay workflows] The fine-tuned model replaces a frontier API call ($0.01–0.05 per classification with GPT-4-class pricing) with a self-hosted inference that costs fractions of a cent per call at volume. At 10,000 replies/month, that's $100–500/month in API savings — and the self-hosted model can be retrained weekly on new labeled data without waiting for a vendor to update their API.

---