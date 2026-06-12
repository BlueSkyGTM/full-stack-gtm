# Phase 05 — NLP Foundations to Advanced (quiz factory)

## Focus

Classic NLP through modern transformers: bag-of-words → embeddings → seq2seq → BERT →
evaluation frameworks. Schema repair is the primary job — all 29 quizzes need trimming
from 8 questions to 6 by removing the extra `pre` and one `post`.

## Scrape hints

- `docs/en.md`: algorithm sections ("How it works"), comparison tables, and
  "When to use / When not to use" boxes are the richest question sources
- `code/main.py`: cite specific function names (e.g. `train_bpe`, `skipgram_pairs`,
  `TextCNN.forward`, `faithfulness`) for Build lessons
- Prerequisites often cite Phase 03 (linear algebra) and Phase 05 earlier lessons;
  only test what this lesson's doc explicitly claims

## Repair pattern (schema_repair)

Current: `pre, pre, check, check, check, post, post, post` (8 questions)
Target: `pre, check, check, check, post, post` (6 questions)

1. Keep the stronger `pre`; drop the other (the one closer to title-guessable).
2. Keep all three `check` questions unchanged — they are correctly structured.
3. Keep 2 of the 3 `post` questions. Prefer the ones that integrate two ideas.
   If no post integrates, rewrite the weakest to combine two doc concepts.
4. For Build lessons: verify at least one check or post names a function from `code/main.*`.

## Style anchor

- `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json` (gold, A-grade)
- `phases/10-llms-from-scratch/34-gradient-checkpointing/quiz.json` (gold, A-grade)

## Common distractor patterns

- Confuse skip-gram vs CBOW direction of prediction
- Confuse BPE merge criterion vs Unigram pruning criterion
- Confuse static embeddings (Word2Vec) vs contextual embeddings (ELMo/BERT)
- Confuse greedy decoding vs beam search tradeoffs
- Confuse faithfulness (hallucination) vs relevance (off-topic) in evaluation
- Confuse encoder-only (BERT) vs decoder-only (GPT) use cases
