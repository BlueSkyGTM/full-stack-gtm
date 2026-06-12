# Phase 04 — Computer Vision (quiz factory)

## Focus

Build-first CV: every lesson implements the algorithm from scratch (NumPy/PyTorch) before
using the library version. Schema repair is the primary job for this phase — all 28 quizzes
need 3 `check` questions added and the double `pre` / triple `post` structure corrected.

## Scrape hints

- `docs/en.md`: "## Build It" section lists the steps to code; use step names and function
  names as hooks for check/post questions
- `code/main.py`: look for function names like `conv2d_naive`, `pad2d`, `im2col`,
  `PatchEmbedding`, `rasterise_2d`, `eval_2d_gaussian`; cite at least one per Build lesson
- Prerequisites span Phases 01–03; only test what this lesson's own doc claims

## Repair pattern (schema_repair)

Current: `pre, pre, post, post, post` (5 questions)
Target: `pre, check, check, check, post, post` (6 questions)

1. Keep the stronger `pre`; drop the other.
2. Write 3 `check` questions — one per distinct doc section. At least one must apply
   a formula or algorithm step (not just name a term).
3. Keep the 2 strongest `post`; drop the third, or rewrite the weakest to integrate
   two doc ideas.
4. Cite a named function from `code/main.*` in at least one check or post.

## Style anchor

- `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json` (gold, A-grade)
- `phases/10-llms-from-scratch/34-gradient-checkpointing/quiz.json` (gold, A-grade)

## Common distractor patterns

- Confuse stride vs padding effect on output dimensions
- Confuse receptive field growth (stacking vs pooling)
- Confuse semantic vs instance segmentation roles
- Confuse transposed convolution with bilinear upsampling
- Confuse ViT patch embedding with standard convolution
- Confuse 3DGS explicit representation with NeRF implicit representation
