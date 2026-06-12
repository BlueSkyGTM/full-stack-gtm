# Phase 07 — Transformers deep dive (quiz factory)

## Focus

Attention, encoders/decoders, efficient attention, and inference tricks built in prior phases.

## Scrape hints

- `docs/en.md`: architecture diagrams, equations named in prose, comparison tables
- `code/main.py`: tensor shapes, mask behavior, forward pass steps
- Prerequisites often cite phase 03–05 and 10; only test what this lesson's doc claims

## Style anchor

- `phases/07-transformers-deep-dive/15-attention-variants/quiz.json` (gold)
- `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json`

## Common distractor patterns

- Full O(N²) vs local/sparse attention complexity
- Encoder-only vs decoder-only vs encoder–decoder roles
- Training objective vs inference optimization (KV cache, speculative decoding)

## Pending in this phase

Regenerate manifest; process all `pending` rows with `"phase": "07"` before moving to phase 08.
