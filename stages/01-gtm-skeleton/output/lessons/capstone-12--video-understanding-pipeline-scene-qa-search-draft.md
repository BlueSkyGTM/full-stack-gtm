# Capstone 12 — Video Understanding Pipeline (Scene, QA, Search)

## Hook

Video is the densest content format most teams handle — a 30-minute sales call contains hundreds of scene transitions, dozens of product mentions, and zero searchable text until you build a pipeline to extract it. This capstone ties frame extraction, visual embedding, and language-based retrieval into one working system.

## Concept

Three mechanisms compose the pipeline: **shot boundary detection** (comparing sequential frame histograms to find scene cuts), **visual question answering** (passing frames + prompts through a multimodal model to extract structured data), and **embedding-based search** (indexing frame descriptions or embeddings for retrieval). Each stage feeds the next — detected scenes become the unit of indexing; QA extracts the metadata; search consumes it.

## Demo

Build a minimal pipeline that: (1) extracts frames from a video at uniform intervals using `ffmpeg`, (2) detects scene boundaries via histogram difference scoring, (3) sends key frames to a vision-language model for description, and (4) stores results in a lightweight searchable index. Every step prints observable output to the terminal — frame counts, boundary timestamps, generated descriptions, and search results.

## Use It

**GTM Redirect:** Sales and success teams record hundreds of hours of product demos and discovery calls. This pipeline maps to **Zone 03 (Content & Personalization)** and **Zone 05 (Enrichment)** — specifically, the ability to chunk video into scenes, tag them by topic, and retrieve "every time the prospect mentioned pricing" or "every demo of the API endpoint." This is the infrastructure behind video-based knowledge bases and call intelligence platforms like Gong or Grain.

**Exercise hooks:**
- **Easy:** Given a directory of pre-extracted frames, compute histogram differences and print timestamps where scene boundaries exceed a configurable threshold.
- **Medium:** Extend the demo pipeline to accept a natural language query (e.g., "show me the whiteboard segment"), embed it, and return the top-3 matching scene descriptions with timestamps.
- **Hard:** Add a QA stage that asks a multimodal model "Is anyone screen-sharing in this frame?" for every scene boundary frame, then index scenes by that binary label. Print a summary of screen-share vs. no-screen-share segment durations.

## Ship It

Write a CLI tool (`video_pipeline.py`) that accepts a video file path and a query string, then outputs scene timestamps with relevance scores. The script must run end-to-end without manual intervention, using `ffmpeg` for extraction and a vision-language model API for description. Include a `--scenes-only` flag that skips QA and just prints shot boundaries.

## Evaluate

Assess whether the pipeline correctly identifies scene transitions on a test video with known cuts, whether QA responses are grounded in frame content vs. hallucinated, and whether search retrieval rank correlates with human-judged relevance. Test edge cases: static videos with few cuts, rapid transitions, and queries that match zero scenes.