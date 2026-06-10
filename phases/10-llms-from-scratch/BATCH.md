# Phase 10 — LLMs From Scratch (quiz factory)

## Focus

Pre-training a transformer LLM end-to-end: tokenisers, data pipeline, distributed training, inference optimisation (KV cache, quantisation, speculative decoding), and the latest attention/architecture research (NSA, differential attention, EAGLE-3).

## Scrape hints

- `docs/en.md`: tokenisation algorithms (BPE, SentencePiece), data pipeline stages, distributed training primitives, inference latency maths, memory budget calculations
- `code/main.py`: pure-Python / numpy implementations — quiz Build lessons on what the code actually demonstrates
- Vocabulary: `glossary/terms.md` for BPE merge, vocab size, activation recomputation, KV cache, quantisation grid

## Style anchor

- `phases/10-llms-from-scratch/15-speculative-decoding-eagle3/quiz.json` — rebuilt gold (EAGLE-3)
- `phases/10-llms-from-scratch/34-gradient-checkpointing/quiz.json` — rebuilt gold (memory tradeoffs)
- `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json` — related lesson for comparison (test the Phase 10 delta, not the Phase 7 foundation)

## Common distractor patterns

- Confuse BPE merge rules with SentencePiece unigram LM probability
- Confuse tensor-parallel (column/row slicing of weight matrices) with pipeline-parallel (layer distribution)
- Confuse KV cache growth rate (linear in context) with attention compute (quadratic)
- Conflate naive full checkpointing (33% overhead) with selective checkpointing (5% overhead)
- Mix EAGLE-1 (hidden-state draft) with EAGLE-3 (TTT, token-prediction objective)

## Sequel-lesson rule

Phase 10 often extends Phase 7 topics (speculative decoding, attention variants). Always test the **delta** — what Phase 10 adds over Phase 7 — not the foundation that Phase 7 already covered. See lesson-planning SKILL.md insight 5 and 6.

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
- Re-test Phase 7 concepts already tested in Phase 7 quizzes for the same lesson family.
