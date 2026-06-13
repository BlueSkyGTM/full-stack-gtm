# LLaVA-OneVision: Single-Image, Multi-Image, Video in One Model

## Hook (Why This Matters)

You've been passing text into LLMs for enrichment and personalization. But prospects' signals are visual—screenshots of their product, logos on their site, demo videos they've published. LLaVA-OneVision is an open multimodal architecture that encodes single images, image sets, and video frames into a shared token space so one model can reason across all three without switching architectures.

## Concept (The Mechanism)

**Single-image encoding** — AnyVision backbone produces visual tokens from one image. These get projected into the language model's embedding space alongside text tokens. This is the same pipeline as LLaVA-1.5.

**Multi-image encoding** — Multiple images are each encoded separately, then their token sequences are concatenated with positional markers indicating which image each token came from. The attention mechanism can cross-reference between images in the same context window.

**Video encoding** — Frames are sampled at uniform intervals, each frame is encoded like a single image, then all frame tokens are concatenated. The model treats video as a very long multi-image sequence with temporal ordering.

**The key mechanism** — One vision encoder, one projection layer, one language model. The difference is how tokens are arranged in the context window, not the architecture itself.

[CITATION NEEDED — concept: LLaVA-OneVision training data mixture ratios for single-image vs. multi-image vs. video tasks]

## Demo (Seeing It Work)

Load LLaVA-OneVision and pass three inputs through the same model:
1. A single screenshot → extract company name and value proposition
2. Two screenshots side-by-side → compare design language
3. A 10-second clip sampled at 3 frames → describe what changes over time

Each run prints the token count breakdown so you can see how the model allocates context budget differently for each modality.

**Exercise hooks:**
- *Easy:* Modify the prompt to extract different fields from the same screenshot.
- *Medium:* Feed two competitor screenshots and ask the model to identify visual differentiation.
- *Hard:* Sample frames from a product demo video at different rates (2fps vs 0.5fps) and compare output consistency.

## Use It (GTM Application)

This is the **Enrichment** cluster. Specifically: processing visual signals that text-only enrichment pipelines miss.

**Single-image use case** — Screenshot a prospect's homepage, extract their positioning language and tech stack indicators. Feed into Clay or similar enrichment waterfall as an additional signal source.

**Multi-image use case** — Pull screenshots from a prospect's last three landing page iterations (via Wayback Machine or manual capture). Ask the model to identify shifts in positioning, new product launches, or pricing changes.

**Video use case** — Process a prospect's recorded product demo. Extract feature names, pricing mentions, competitive callouts. Use as pre-call intelligence.

**Exercise hooks:**
- *Easy:* Write a single-image prompt that extracts company name, industry, and one value proposition from a screenshot.
- *Medium:* Build a function that takes a list of image URLs, encodes them as a multi-image input, and returns a comparison table.
- *Hard:* Write a script that samples frames from a YouTube demo video at regular intervals, passes them through LLaVA-OneVision, and outputs a structured feature list with timestamps.

## Ship It (Production Deployment)

**Context window budgeting** — Video frames eat tokens fast. At 24 visual tokens per frame and 30 frames per video, you're at 720 tokens for visual input alone. Set hard limits on frame sampling rates.

**Latency reality** — Multi-image and video inputs have linear latency scaling with frame count. Benchmark single-image vs. 5-frame video to establish your SLA thresholds.

**Caching strategy** — If you're processing the same company's homepage screenshot weekly, cache the visual tokens. Re-encode only when the image hash changes.

**Fallback chain** — If LLaVA-OneVision fails on a video input (timeout, context overflow), fall back to sampling 3 key frames and treating it as a multi-image task.

[CITATION NEEDED — concept: LLaVA-OneVision recommended inference hardware requirements for video inputs]

**Exercise hooks:**
- *Easy:* Add token counting to your inference call and log it alongside the response.
- *Medium:* Implement a frame sampling function that caps total visual tokens at a configurable budget.
- *Hard:* Build a caching layer that stores visual token embeddings keyed by image hash, with a TTL-based invalidation strategy.

## Debug It (Common Failure Modes)

**Mode confusion** — The model sometimes treats a multi-image input as a single composite image. Symptom: responses about "the image" when you sent three. Fix: explicitly label each image in the text prompt ("Image 1 shows X, Image 2 shows Y").

**Temporal hallucination on video** — When sampled frames don't capture key transitions, the model fabricates what happened between frames. Symptom: confident descriptions of events that aren't in the video. Fix: increase frame sampling rate for videos with fast cuts, or ask the model to report confidence per frame.

**Resolution sensitivity** — Company logos with small text become illegible when downsampled to the model's input resolution. Symptom: wrong company names. Fix: pre-crop the logo region at higher resolution before passing to the model.

**Context overflow on long videos** — Exceeding the context window causes silent truncation of later frames. Symptom: the model only describes the first half of a video. Fix: log frame count and estimated token count before inference; reject or split inputs that exceed your configured limit.

**Exercise hooks:**
- *Easy:* Reproduce the mode confusion failure by sending two images without labels, then fix it by adding explicit labels.
- *Medium:* Create a video input that triggers temporal hallucination (fast transitions sampled at low fps), then adjust the sampling rate until the output stabilizes.
- *Hard:* Build a pre-flight check function that estimates total token count from frame count and resolution, rejects inputs above threshold, and logs the rejection reason.