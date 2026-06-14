## Ship It

Deploy the extraction recipe behind a FastAPI endpoint so any tool in your enrichment stack — Clay, n8n, a Python script, a Zapier webhook — can call VLM extraction over HTTP without depending on closed APIs. The endpoint includes a startup health check (model loaded, VRAM confirmed), batched inference for throughput, and an OOM fallback that drops to a smaller quantized model if the primary allocation fails.

Zone 12 observability hooks into this endpoint at two points. First, the `/health` endpoint reports model status, VRAM usage, and a rolling extraction success rate — when success rate drops below a threshold, the tracing setup flags it as model degradation before it silently corrupts your enrichment data. Second, every `/extract` call emits a structured log line with the same fields as the `ExtractionLog` dataclass above, enabling you to trace any downstream reply-rate regression back to the specific extraction call that produced bad data.

```python
import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict

import httpx
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vlm-extractor")

class ExtractRequest(BaseModel):
    image_url: str
    fields: dict = {"company_name": "", "industry": "", "tagline": ""}
    max_tokens: int = 256

class ExtractResponse(BaseModel):
    extracted: dict | None
    raw_output: str
    model_used: str
    quantization: str
    json_valid: bool
    latency_ms: int

stats = {
    "requests_total": 0,
    "json_valid_count": 0,
    "fallback_count": 0,
    "recent_results": deque(maxlen=100)
}

PRIMARY_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
model_state = {"model": None, "processor": None, "model_id": None, "quant": None}

def load_model(model_id, quantization=None):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    kwargs = {"trust_remote_code": True}
    if quantization == "4bit":
        kwargs["load_in_4bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading primary model...")
    try:
        model, processor = load_model(PRIMARY_MODEL)
        model_state.update({"model": model, "processor": processor, "model_id": PRIMARY_MODEL, "quant": "fp16"})
        logger.info(f"Loaded {PRIMARY_MODEL} at fp16")
    except Exception as e:
        logger.error(f"Primary model load failed: {e}")
        try:
            model, processor = load_model(FALLBACK_MODEL, "4bit")
            model_state.update({"model": model, "processor": processor, "model_id": FALLBACK_MODEL, "quant": "4bit"})
            logger.warning(f"Fallback to {FALLBACK_MODEL} at 4bit")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            sys.exit(1)
    
    vram_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    logger.info(f"VRAM allocated: {vram_alloc:.2f} GB")
    yield
    logger.info("Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    vram_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    vram_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    recent = list(stats["recent_results"])
    success_rate = sum(recent) / len(recent) if recent else 0.0
    return {
        "status": "healthy" if model_state["model"] else "degraded",
        "model_loaded": model_state["model_id"],
        "quantization": model_state["quant"],
        "vram_allocated_gb": round(vram_alloc, 2),
        "vram_reserved_gb": round(vram_reserved, 2),
        "requests_total": stats["requests_total"],
        "json_valid_rate": round(success_rate, 3),
        "fallback_count": stats["fallback_count"],
        "degradation_warning": success_rate < 0.85 and len(recent) >= 20
    }

@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    start = time.time()
    stats["requests_total"] += 1
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(req.image_url, timeout=30)
        if resp.status_code != 200:
            raise HTTPException(400, f"Image fetch failed: {resp.status_code}")
    
    from io import BytesIO
    image = Image.open(BytesIO(resp.content)).convert("RGB")
    
    prompt = f"""Extract structured data from this image.
Return ONLY valid JSON with these fields: {json.dumps(req.fields)}
If a field is not visible, return null. Do not guess."""
    
    model = model_state["model"]
    processor = model_state["processor"]
    
    try:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM on primary model, falling back...")
        stats["fallback_count"] += 1
        del model_state["model"]
        torch.cuda.empty_cache()
        model, processor = load_model(FALLBACK_MODEL, "4bit")
        model_state.update({"model": model, "processor": processor, "model_id": FALLBACK_MODEL, "quant": "4bit"})
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=False)
        generated = output_ids[:, inputs["input