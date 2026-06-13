# Keypoint Detection & Pose Estimation

## Hook It
Run a single-frame pose estimator on an image and print detected joint coordinates to stdout. The practitioner sees a skeleton materialize from raw pixel data in under 20 lines of code.

## Ground It
Define what a keypoint is (an annotated (x, y, confidence) tuple tied to a semantic landmark), what a skeleton topology is (the graph connecting keypoints into a body model), and the two dominant architectures: heatmap regression (learn a probability surface per joint, take argmax) versus coordinate regression (directly output (x, y) per joint). Introduce the COCO 17-keypoint format as the de facto standard.

## Explain It
Walk through the mechanism of bottom-up pose estimation (detect all keypoints, then associate into skeletons via part affinity fields or tagging) versus top-down (detect persons first, then estimate keypoints per cropped bounding box). Compare throughput tradeoffs: bottom-up runs once per image regardless of person count; top-down scales linearly with persons. Cover confidence filtering, keypoint occlusion handling, and how spatial temperature in heatmap generation affects localization precision. Introduce MediaPipe's graph-based pipeline as an implementation of the bottom-up pattern with on-device constraints.

## Use It
GTM redirect: foundational for **Zone 2 — Intent Signals**. Pose estimation enables detection of engagement gestures (leaning in, hand raises, head orientation) in recorded sales calls and webinar footage. The practitioner builds a function that ingests a video frame and emits a structured JSON object of keypoint coordinates and a binary "speaker facing camera" flag. This output feeds downstream into attention-scoring heuristics used by conversation intelligence platforms. [CITATION NEEDED — concept: conversation intelligence pose/gesture scoring in GTM tools]

## Ship It
Package a CLI tool that accepts a video file path, samples frames at 1 fps, runs MediaPipe Pose, and writes a JSONL file where each line contains `{timestamp, keypoints: [{name, x, y, confidence}], facing_camera: bool}`. The practitioner runs it on a recorded Zoom call and confirms output by printing the first 3 lines.

## Prove It

- **Easy**: Write a function that takes MediaPipe Pose landmarks and returns the pixel distance between left wrist and right wrist, printing the result for a single image.
- **Medium**: Implement a heuristic that classifies a frame as "speaker facing camera" based on nose-to-ear symmetry (left ear vs. right ear visibility confidence delta below a threshold), and test it on 5 frames printing the classification per frame.
- **Hard**: Build the full CLI pipeline described in Ship It — video in, JSONL out — with a `--min-confidence` flag that filters keypoints below the threshold, and confirm total keypoint count changes when the flag is adjusted.