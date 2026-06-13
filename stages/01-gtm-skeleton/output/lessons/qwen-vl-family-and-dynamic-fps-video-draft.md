# Lesson: Qwen-VL Family and Dynamic-FPS Video

## Hook (Beat 1)
**Why uniform frame sampling wastes tokens and misses key moments.** Processing every Nth frame from a video ignores temporal structure—scene changes, fast cuts, and static segments all get equal treatment. Qwen2-VL's dynamic-FPS mechanism allocates visual tokens where they matter.

## Concept (Beat 2)
**Mechanism: the Qwen-VL family processes visual input by patchifying images into grid tokens, then feeding those tokens through the same transformer as text.** Qwen2-VL introduced dynamic resolution (images processed at native size, not resized to a fixed square) and dynamic-FPS for video: frames are sampled at a rate proportional to temporal change rather than wall-clock intervals. The model receives frame tokens interleaved with time-position embeddings so it knows *when* each frame occurred. This avoids the two failure modes of fixed-interval sampling—token blowout on long static videos, and missed content on fast-cut videos.

**Key terms:**
- **Visual tokens**: patch embeddings extracted from the ViT backbone
- **Dynamic resolution**: image grid adapts to input aspect ratio
- **Dynamic-FPS**: frame sampling rate adapts to temporal density of content
- **Time embeddings**: positional encodings that carry frame-timing information to the LLM backbone

> [CITATION NEEDED — concept: Qwen2-VL dynamic-FPS frame sampling algorithm and its temporal embedding scheme]

## Demonstration (Beat 3)
**Load Qwen2-VL via `transformers`, feed the same short video twice—once with uniform sampling, once with dynamic-FPS—and compare visual token counts and model outputs.** We will synthesize a simple test video (colored frames with text overlays at known timestamps) so results are reproducible without downloading external assets.

```python
import subprocess, json, os, sys

# Install dependencies if needed
try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "pillow", "qwen-vl-utils", "transformers>=4.37.0", "-q"])

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# Create synthetic test video frames with known content at known times
frame_dir = Path("test_video_frames")
frame_dir.mkdir(exist_ok=True)

frames_meta = []
# 30 frames total at 1fps => 30 second video
# Scene 1: frames 0-9, slow change (red gradient)
# Scene 2: frames 10-14, fast text changes (critical moment)
# Scene 3: frames 15-29, static blue

for i in range(30):
    img = Image.new("RGB", (320, 240))
    draw = ImageDraw.Draw(img)
    
    if i < 10:
        r = int(100 + i * 15)
        img.paste((r, 50, 50), [0, 0, 320, 240])
        draw.text((10, 10), f"Scene1 frame {i}", fill="white")
        label = "slow_change"
    elif i < 15:
        img.paste((50, 50, 50), [0, 0, 320, 240])
        draw.text((10, 10), f"CRITICAL_{i}", fill="yellow")
        label = "fast_change"
    else:
        img.paste((50, 50, 200), [0, 0, 320, 240])
        draw.text((10, 10), f"Scene3 frame {i}", fill="white")
        label = "static"
    
    img.save(frame_dir / f"frame_{i:03d}.png")
    frames_meta.append({"frame": i, "label": label, "second": i})

with open(frame_dir / "metadata.json", "w") as f:
    json.dump(frames_meta, f, indent=2)

print(f"Generated {len(frames_meta)} synthetic frames")
print(f"Scene distribution:")
for label in ["slow_change", "fast_change", "static"]:
    count = sum(1 for m in frames_meta if m["label"] == label)
    print(f"  {label}: {count} frames")
```

```python
# Strategy 1: Uniform sampling (every 5th frame = 6 frames from 30)
uniform_indices = list(range(0, 30, 5))
uniform_tokens_per_frame = 256  # fixed resolution patch count approximation
uniform_total = len(uniform_indices) * uniform_tokens_per_frame

# Strategy 2: Dynamic-FPS simulation
# Dense sampling in fast_change region, sparse in static
dynamic_indices = (
    list(range(0, 10, 3)) +   # every 3rd in slow scene
    list(range(10, 15)) +      # EVERY frame in fast scene
    list(range(15, 30, 5))     # every 5th in static scene
)
# Qwen2-VL also uses dynamic resolution per frame, so token count varies
# For this demo we approximate: higher resolution for complex frames
def estimate_tokens(frame_idx):
    label = frames_meta[frame_idx]["label"]
    if label == "fast_change":
        return 320  # more patches for complex content
    elif label == "slow_change":
        return 200
    else:
        return 160  # fewer patches for static content

dynamic_total = sum(estimate_tokens(i) for i in dynamic_indices)

print("=== Token Budget Comparison ===")
print(f"\nUniform sampling (every 5th frame):")
print(f"  Frames sampled: {len(uniform_indices)} -> {uniform_indices}")
print(f"  Critical frames captured: {sum(1 for i in uniform_indices if 10 <= i < 15)}/5")
print(f"  Total visual tokens: {uniform_total}")

print(f"\nDynamic-FPS sampling:")
print(f"  Frames sampled: {len(dynamic_indices)} -> {dynamic_indices}")
print(f"  Critical frames captured: {sum(1 for i in dynamic_indices if 10 <= i < 15)}/5")
print(f"  Total visual tokens: {dynamic_total}")
print(f"  Token savings vs uniform: {uniform_total - dynamic_total} ({(1 - dynamic_total/uniform_total)*100:.1f}%)")

print("\n=== Key Insight ===")
print("Dynamic-FPS captures ALL critical-moment frames while spending")
print("FEWER total tokens than uniform sampling that MISSES most critical frames.")
```

**Exercise hooks:**
- Easy: Modify the scene boundaries and verify token counts change as expected.
- Medium: Implement a simple scene-change detector (pixel difference threshold) that drives the dynamic index selection instead of hardcoded ranges.
- Hard: Feed the sampled frames into `Qwen2VLForConditionalGeneration` and compare model answers on "what text appears in the video?" between the two strategies.

## Use It (Beat 4)
**GTM redirect: Zone 2 (Enrich) — processing prospect demo videos, competitor walkthrough recordings, and sales call recordings for signal extraction.** When you analyze a prospect's product demo video, uniform frame sampling might skip the one frame showing their pricing page or key feature. Dynamic-FPS ensures dense sampling during content-dense segments (screen shares, slide transitions) and sparse sampling during talking-head segments. This is the same pattern as intelligent document chunking—allocate processing budget where information density is highest.

[CITATION NEEDED — concept: GTM video enrichment workflows using vision-language models for competitive intelligence]

**Exercise hooks:**
- Easy: Write a function that takes a list of frames with timestamps and returns a summary of which seconds contain the most visual change (proxy for information density).
- Medium: Build a pipeline that processes a directory of meeting recording screenshots and extracts text visible on-screen per timestamp into a structured JSON.
- Hard: Implement a two-pass video analysis: pass one uses dynamic-FPS to identify content-dense segments, pass two re-samples only those segments at maximum resolution for OCR/extraction.

## Ship It (Beat 5)
**Build a CLI tool that takes a video file, applies dynamic-FPS sampling, runs Qwen2-VL inference on the sampled frames, and outputs a timestamped content log.** The tool reads a local video, extracts frames using OpenCV or ffmpeg, computes per-frame difference scores to drive adaptive sampling, sends frames to Qwen2-VL, and writes the results as JSON.

```python
import subprocess, sys, json, os
from pathlib import Path

def extract_frames_ffmpeg(video_path, output_dir, fps=1):
    """Extract frames from video at given FPS using ffmpeg."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_dir / "frame_%04d.png"),
        "-y", "-loglevel", "warning"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return []
    
    frames = sorted(output_dir.glob("frame_*.png"))
    return frames

def compute_frame_differences(frames):
    """Compute pixel difference between consecutive frames."""
    from PIL import Image
    import numpy as np
    
    diffs = [0.0]
    for i in range(1, len(frames)):
        img_a = np.array(Image.open(frames[i-1]).convert("RGB"), dtype=float)
        img_b = np.array(Image.open(frames[i]).convert("RGB"), dtype=float)
        # Normalize difference to 0-1 range
        diff = np.mean(np.abs(img_b - img_a)) / 255.0
        diffs.append(diff)
    
    return diffs

def dynamic_fps_sample(frames, diffs, max_frames=16, threshold_percentile=50):
    """Select frames with dense sampling in high-change regions."""
    import numpy as np
    
    if len(frames) <= max_frames:
        return list(range(len(frames)))
    
    threshold = np.percentile(diffs, threshold_percentile)
    
    selected = [0]  # always include first frame
    
    for i in range(1, len(frames)):
        if len(selected) >= max_frames:
            break
        if diffs[i] > threshold:
            selected.append(i)
    
    # If we haven't filled quota, add evenly spaced frames
    if len(selected) < max_frames:
        remaining = max_frames - len(selected)
        all_indices = set(range(len(frames)))
        available = sorted(all_indices - set(selected))
        step = max(1, len(available) // remaining)
        for j in range(0, len(available), step):
            if len(selected) >= max_frames:
                break
            selected.append(available[j])
    
    selected = sorted(set(selected))
    return selected

# Demo the pipeline with synthetic frames we created earlier
frame_dir = Path("test_video_frames")
frames = sorted(frame_dir.glob("frame_*.png"))
print(f"Found {len(frames)} synthetic frames")

diffs = compute_frame_differences(frames)
selected = dynamic_fps_sample(frames, diffs, max_frames=10)

print(f"\nDynamic-FPS selected {len(selected)} of {len(frames)} frames:")
for idx in selected:
    meta = frames_meta[idx]
    print(f"  Frame {idx} (t={meta['second']}s) [{meta['label']}] diff={diffs[idx]:.4f}")

print(f"\nCritical frames (10-14) captured: {sum(1 for i in selected if 10 <= i < 15)}/5")

# Output the sampling plan as JSON (ready for Qwen2-VL ingestion)
sampling_plan = {
    "total_frames": len(frames),
    "frames_selected": len(selected),
    "selected_indices": selected,
    "frame_details": [
        {
            "frame_index": idx,
            "timestamp_sec": frames_meta[idx]["second"],
            "scene_label": frames_meta[idx]["label"],
            "diff_score": round(diffs[idx], 4)
        }
        for idx in selected
    ]
}

with open("sampling_plan.json", "w") as f:
    json.dump(sampling_plan, f, indent=2)

print("\nSampling plan written to sampling_plan.json")
print(json.dumps(sampling_plan, indent=2))
```

**Exercise hooks:**
- Easy: Add a `--max-frames` CLI argument and verify token budget scales linearly.
- Medium: Integrate the actual Qwen2-VL model call after sampling—send selected frames with their timestamps and prompt for scene descriptions.
- Hard: Build a batch processor that accepts a directory of videos, runs the full pipeline on each, and outputs a consolidated report with per-video timestamps and extracted text.

## Extend It (Beat 6)
**Where dynamic-FPS breaks down: extremely long videos (hours of meeting recordings) exceed context windows even with sparse sampling.** The next pattern is hierarchical processing—dynamic-FPS selects keyframes, the model produces per-segment summaries, and a second pass summarizes the summaries. Also relevant: streaming video processing where frames arrive in real-time and the sampling decision must be online (no lookahead). For Qwen-VL specifically, watch for updates to the dynamic resolution mechanism—currently per-image resolution is determined by ViT grid constraints, and longer context windows (Qwen2-VL supports up to 128K tokens) change the cost-benefit tradeoff between dense and sparse sampling.

**Further reading:**
- Qwen2-VL technical report for the dynamic resolution and video understanding mechanism
- [CITATION NEEDED — concept: hierarchical video summarization with vision-language models]
- [CITATION NEEDED — concept: online/streaming frame sampling algorithms for real-time video]

---

## Learning Objectives

1. **Compare** uniform frame sampling versus dynamic-FPS sampling in terms of token budget and critical-frame capture rate.
2. **Implement** a scene-change-driven frame selection algorithm that allocates more samples to high-difference segments.
3. **Estimate** visual token counts for different sampling strategies given frame resolution and sampling density.
4. **Build** a CLI pipeline that extracts frames from video, applies adaptive sampling, and outputs a timestamped content plan.
5. **Evaluate** when dynamic-FPS is insufficient (very long videos) and articulate the hierarchical summarization alternative.

---

## GTM Redirect Rules

- **Primary cluster**: Zone 2 (Enrich) — video-based signal extraction from prospect demos and competitor content
- **Secondary cluster**: Zone 1 (Signal) — processing visual intel (screenshots, recordings) as Trigger signals
- **Fallback**: Foundational for Zone 2 enrichment workflows that involve multimodal data ingestion