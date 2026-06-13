# Open-Weight VLM Recipes: What Actually Matters

## Hook
Three open-weight VLMs score within 2% on MMMU. One extracts structured data from your screenshots reliably. The other two hallucinate field values. Benchmark ceilings hide the failure modes that matter for production pipelines.

## Learn It
Vision-Language Models differ on three axes that benchmarks obscure: vision encoder resolution and preprocessing, the connector architecture (linear projection vs. Q-Former vs. pixel shuffle), and training data composition. Compare LLaVA-OneVision, Qwen2-VL, and InternVL3 on these axes. Mechanism: resolution handling determines whether small text in screenshots is legible to the model; connector type determines how much vision information survives compression into the LLM's token space.

## See It
Run the same structured extraction prompt across three models using `transformers` — a company logo screenshot parsed into `{company_name, industry, tagline}`. Print each model's raw output, token-by-token, to show where hallucination starts. Demonstrate that quantization (4-bit vs. 8-bit) degrades OCR-reliant extraction before it degrades scene-description quality. Observable output: raw text, extracted fields, pass/fail per model per quant level.

## Use It
GTM redirect: screenshot-to-structured-data extraction feeds the enrichment waterfall in [CITATION NEEDED — concept: VLM-based enrichment in Clay waterfall]. Build a recipe that takes a list of website screenshot URLs and returns normalized company data. The recipe selects model + quantization based on a task classifier: OCR-heavy tasks get full-precision Qwen2-VL at higher resolution; scene-description tasks can use quantized LLaVA-OneVision at standard resolution. Print the routing decision and extraction result for each input.

**Exercise hooks:**
- *Easy:* Run a single image through two models and diff the JSON output.
- *Medium:* Build the task-classifier routing logic that picks model + quant based on image characteristics.
- *Hard:* Evaluate extraction accuracy across 20 screenshots with varying text density; plot failure rate vs. estimated text pixel height.

## Ship It
Deploy a quantized Qwen2-VL-7B-Instruct behind a FastAPI endpoint that accepts image URLs and returns structured JSON. Include: startup health check (model loaded, VRAM confirmed), batched inference for throughput, and a fallback to a smaller model if VRAM is insufficient. This endpoint plugs directly into enrichment workflows — any tool that can call an HTTP endpoint can now run VLM extraction without depending on closed APIs.

**Exercise hooks:**
- *Easy:* Wrap a single-model inference call in a `/extract` endpoint and curl it.
- *Medium:* Add batched inference, quantization auto-selection, and `/health`.
- *Hard:* Add retry logic that falls back to a smaller model on OOM, logs the fallback event, and maintains a rolling accuracy estimate.

## Prove It
Build a tiny evaluation harness: 20 labeled screenshots (logos, landing pages, pricing tables), expected output schema, and three metrics — field-level exact match, field-level fuzzy match (Levenshtein ≤ 2), and hallucination rate (fields present in output but absent in image). Run all three models through the harness. Print a comparison table. The practitioner leaves knowing which model-quantization combo works for *their* task, not which one tops a leaderboard.

**Exercise hooks:**
- *Easy:* Run the eval harness on one model and print the three metrics.
- *Medium:* Run all three models, print the comparison table, identify the winner for OCR-heavy vs. scene-description tasks.
- *Hard:* Add a fourth metric — confidence calibration (does the model's stated confidence correlate with actual accuracy?) — and report which model is best calibrated.