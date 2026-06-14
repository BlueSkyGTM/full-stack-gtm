## Ship It

The production version takes a CSV of prospects, generates one video per row, and writes an output CSV mapping company to video URL and full generation config. This is the same pattern as batch image generation for social content — same content scaling cluster, same pipeline structure — but the output is a video asset that gets embedded in outreach sequences rather than posted to social.

```python
import csv
import json
import replicate
import os
import time

INPUT_CSV = "prospects.csv"
OUTPUT_CSV = "video_outreach_results.csv"
PROMPT_TEMPLATE = "A short cinematic establishing shot of a {industry} company environment, soft natural light, slow subtle camera movement, professional and aspirational, high quality"

def load_prospects(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def generate_clip(prompt, seed, motion_bucket_id, fps, video_length):
    output = replicate.run(
        "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5bcccg7e8a8b6f",
        input={
            "image": "https://replicate.delivery/mgxt/brand_seed.jpg",
            "motion_bucket_id": motion_bucket_id,
            "fps": fps,
            "seed": seed,
            "video_length": video_length,
            "cond_aug": 0.02
        }
    )
    url = output if isinstance(output, str) else output[0] if isinstance(output, list) else str(output)
    return url

prospects = load_prospects(INPUT_CSV)
print(f"Loaded {len(prospects)} prospects from {INPUT_CSV}\n")

results = []

for i, row in enumerate(prospects):
    company = row["company"]
    industry = row.get("industry", "corporate")
    prompt = PROMPT_TEMPLATE.format(industry=industry)
    config = {
        "seed": 42 + i,
        "motion_bucket_id": int(row.get("motion_bucket_id", 40)),
        "fps": 8,
        "video_length": 14
    }
    print(f"[{i+1}/{len(prospects)}] Generating for {company}...")
    print(f"  Prompt: {prompt}")
    print(f"  Config: {json.dumps(config)}")

    try:
        video_url = generate_clip(prompt, **config)
        status = "success"
        print(f"  URL: {video_url}\n")
    except Exception as e:
        video_url = ""
        status = f"error: {str(e)}"
        print(f"  FAILED: {e}\n")

    results.append({
        "company": company,
        "industry": industry,
        "video_url": video_url,
        "status": status,
        "seed": config["seed"],
        "motion_bucket_id": config["motion_bucket_id"],
        "fps": config["fps"],
        "video_length": config["video_length"],
        "prompt": prompt
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Wrote {len(results)} rows to {OUTPUT_CSV}")
```

This script requires a `prospects.csv` with at minimum a `company` column and optional `industry` and `motion_bucket_id` columns. The output CSV is designed to be imported back into a CRM or sequencer: the `video_url` column maps to a custom field, and the config columns provide auditability for which parameters produced which asset. The fixed seed base (42 + index) ensures that re-running the pipeline with the same CSV reproduces the same clips — critical for debugging and for compliance when a prospect asks what they were shown.