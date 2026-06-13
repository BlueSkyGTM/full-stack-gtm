# Video-Language Models: Temporal Tokens and Grounding

## Hook

Video is not a stack of images. A 60-second product demo contains ~1,800 frames at 30fps, but the moment the prospect says "that's the feature I need" occupies maybe 90 frames. Temporal grounding is the mechanism that pins language to those 90 frames — and without it, you're averaging signal into noise.

## Concept

**Temporal tokens** extend the vision transformer paradigm into the time dimension. Where a ViT produces spatial patch tokens from a single image, a video-language model must produce spatiotemporal tokens from a frame sequence. Three dominant mechanisms: (1) factorized attention — attend spatially within a frame, then temporally across frames (TimeSformer); (2) 3D patch embedding — treat time as a third convolution dimension (ViViT); (3) frame-level pooling with per-frame CLS tokens stitched into a temporal sequence. Each trades compute for temporal resolution differently.

**Temporal grounding** is the retrieval inverse: given a text query ("when does the speaker mention pricing?"), return the start and end timestamp in the video. This is structured as span prediction over the temporal token sequence — the same head architecture as extractive QA (start/end logits over a token sequence), but the tokens represent time slices instead of words. [CITATION NEEDED — concept: MAD benchmark, ActivityNet Grounding results for current SOTA]

## Apply

Build a temporal grounding pipeline on a short video segment. Extract frame-level embeddings using a pretrained vision encoder, construct a temporal token sequence, implement a simple start/end span predictor over that sequence, and verify the predicted timestamp matches the known answer. Print frame indices, confidence scores, and the predicted time span.

**Exercise hooks:**
- Easy: Given pre-extracted frame embeddings for a 10-second clip, write the span prediction head and locate the frame closest to a text query embedding.
- Medium: Implement factorized spatial-then-temporal attention on a 16-frame video clip. Print attention weights to confirm temporal heads attend across frames, not just within.
- Hard: Build a sliding-window temporal grounding system over a 5-minute video. Handle overlapping predictions and merge adjacent high-confidence spans.

## Use It

**GTM Cluster: Zone 2 — Signal Intelligence / Conversation Analysis**

Temporal grounding maps directly to call recording and demo analysis. A 45-minute Zoom recording is a video. The question "when did the champion mention budget approval?" is a temporal grounding query. The mechanism is identical: encode the audio-visual stream into temporal tokens, project the text query into the same space, predict the time span.

This is the retrieval-augmented generation pattern applied to video: instead of chunking a document into text passages, you chunk a recording into temporal windows and ground language queries against them. Tools like Gong and Chorus implement proprietary versions of this; the underlying mechanism is span prediction over temporal tokens.

**Exercise hook:** Write a pseudo-grounding pipeline that takes a transcript with timestamps (simulating ASR output from a recorded demo call) and locates the 30-second window where the prospect discusses integration requirements. Print the timestamp range and confidence.

## Ship It

Package the temporal grounding logic from Apply into a reusable module: input is a video file path and a list of text queries, output is a JSON object mapping each query to its predicted `{start_seconds, end_seconds, confidence}`. Handle edge cases: queries with no valid span (confidence below threshold), overlapping spans, and videos shorter than the model's context window. Write a CLI entry point that prints the JSON to stdout.

**Exercise hook:** Build the full pipeline end-to-end: video → frame extraction → temporal token encoding → grounding inference → JSON output. Test against a publicly available video with known ground-truth annotations (e.g., a segment from ActivityNet QA). Print the full JSON result.

## Extend

- **Dense video captioning**: Instead of grounding a single query, generate natural language descriptions for every temporal segment in the video. This is temporal grounding + generation, not just retrieval. Models like ViTT and PandaGPT implement this.
- **Streaming temporal grounding**: Process video frames as they arrive instead of batch-processing the full clip. This requires causal temporal attention — no looking ahead at future frames.
- **Multi-query grounding at scale**: Index thousands of videos and ground queries across the entire corpus. This is vector search over temporal tokens — same pattern as RAG, but the chunks are video segments, not text passages. [CITATION NEEDED — concept: production architectures for multi-video temporal grounding at scale]

---

**Learning Objectives (draft):**

1. Compare three mechanisms for constructing temporal tokens from video frame sequences (factorized attention, 3D patch embedding, frame-level pooling).
2. Implement a temporal grounding span predictor that returns start/end frame indices given a text query and a sequence of frame embeddings.
3. Evaluate grounding predictions against ground-truth timestamps using IoU (Intersection over Union) over temporal spans.
4. Configure a sliding-window inference pipeline for videos longer than the model's temporal context window.
5. Map the temporal grounding mechanism to GTM applications in call recording and demo analysis.