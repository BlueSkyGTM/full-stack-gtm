# Long-Video Understanding at Million-Token Context

## Beat 1: Hook — Why Million-Token Video Matters

A 2-hour webinar at 1 fps produces ~7,200 frames. Each frame tokenizes to hundreds of visual tokens. You either chunk and lose temporal coherence, or you need a context window that swallows the entire sequence. Million-token context windows make single-pass video reasoning possible — but the naive approach fails for reasons that have nothing to do with the model and everything to do with how video becomes tokens.

## Beat 2: Concept — The Frame-to-Token Pipeline

Explain the encoding chain: raw video → frame sampling → image encoding (SigLIP/ViT) → patch tokenization → token sequence with temporal position embeddings. Cover the three sampling regimes (uniform, scene-boundary, keyframe) and their token budgets. Establish the math: at 257 tokens per frame (SigLIP-SO400M), 7,200 frames = 1.85M tokens — already over budget before you add text. The constraint is real, and frame selection strategy determines what the model can "see."

## Beat 3: Mechanism — How Long-Context Attention Actually Works

Explain the attention patterns that make million-token video tractable: ring attention, sparse attention blocks, and temporal locality bias. Cover how models like Gemini 1.5 Pro and Qwen2-VL handle the memory wall — interleaved frame/text compression, sliding-window attention over visual tokens, and the difference between "the model saw all frames" and "the model can retrieve any frame equally well." End with the degradation curve: retrieval accuracy over temporal distance at different context lengths per published benchmarks. [CITATION NEEDED — concept: temporal retrieval degradation at 1M tokens for video QA benchmarks]

## Beat 4: Code — Build a Frame Budget Calculator and Test It

Working Python script that takes a video file path, extracts frames at configurable fps, runs token count estimation, and prints a budget report: total frames, estimated tokens, headroom against context limits (1M, 2M), and recommended sampling strategy. No API keys required — uses OpenCV for frame extraction and the known token-per-frame constants. Exercise hook: Easy — run the calculator on a local video. Medium — modify to auto-select sampling fps to fit a target context window. Hard — implement scene-boundary detection (OpenCV histogram diff) and compare token budgets against uniform sampling.

## Beat 5: Use It — GTM Application: Full-Webinar Competitive Intelligence

This maps to **Signal Capture → Competitive Intelligence** in the GTM topic map. The mechanism: ingest a competitor's full product demo webinar (60-120 min) in a single pass, then query for feature claims, pricing language, positioning shifts, and objection handling patterns. Single-pass temporal reasoning means you can ask "what did they say about pricing after the security question at minute 47?" and the model retains both the content and the sequence. Exercise hook: Easy — write a prompt that extracts a structured feature comparison table from a long video transcript. Medium — build a pipeline that takes a YouTube URL, downloads the video, samples frames to fit a 1M token budget, and sends to a VLM API with a competitive intelligence extraction prompt. Hard — compare outputs from full-video ingestion vs. chunked transcript-only analysis on the same source; measure which approach captures temporal dependencies (e.g., "they contradicted the earlier claim").

## Beat 6: Ship It — Production Constraints and Failure Modes

Deploying long-video understanding in production hits three walls: cost (1M input tokens is expensive per query), latency (prefill over million tokens takes minutes, not seconds), and the recall cliff (models reliably retrieve from the first and last ~20% of the context; the middle is a dead zone for details). Cover mitigation strategies: pre-indexing video segments with embeddings for RAG-based retrieval before the long-context call, two-pass architectures (fast scan → targeted deep read), and the hard truth that for many use cases, transcript + selective frame sampling outperforms full-video ingestion. Exercise hook: Easy — benchmark time-to-first-token at 100K vs 500K vs 1M tokens using an API endpoint. Medium — implement a two-pass pipeline: dense transcript search to identify key timestamps, then targeted frame extraction at those timestamps only. Hard — build an evaluation harness that measures factual recall at 10 evenly-spaced points in a long video, plot the temporal recall curve, and identify where the model's memory fails.

---

**Learning Objectives (draft):**
1. Calculate token budgets for video inputs given frame rate, resolution, and context window constraints.
2. Compare uniform, keyframe, and scene-boundary sampling strategies and their impact on temporal coverage.
3. Implement a frame budget calculator that auto-selects sampling parameters to fit a target context window.
4. Evaluate retrieval accuracy across temporal positions in a long video sequence.
5. Deploy a two-pass video analysis pipeline that combines transcript-based retrieval with targeted frame extraction.