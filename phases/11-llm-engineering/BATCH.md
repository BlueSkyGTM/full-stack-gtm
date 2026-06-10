# Phase 11 — LLM Engineering (quiz factory)

## Focus

Production LLM usage patterns: prompt engineering, RAG pipelines, structured outputs, function calling, fine-tuning (LoRA, QLoRA), inference serving (vLLM, batching strategies), and LangChain / LangGraph orchestration.

## Scrape hints

- `docs/en.md`: focus on the engineering tradeoff sections, latency vs throughput diagrams, LoRA parameter-count maths, and RAG pipeline stage descriptions
- `code/main.py`: API usage patterns, schema definitions — quiz what the code demonstrates end-to-end
- Vocabulary: `glossary/terms.md` for LoRA rank, quantisation, continuous batching, retrieval pipeline, structured output

## Phase 11 batch note

All Phase 11 rows in the manifest are `fill_explanations` jobs — quiz files already exist and have valid questions but empty `explanation` fields. Your job is to write 1–3 sentence explanations for each question using the lesson's `docs/en.md` as source. Do not change the question text, options, or `correct` index.

## Style anchor

- No gold quiz produced yet in Phase 11 — use `phases/07-transformers-deep-dive/15-attention-variants/quiz.json` as a depth benchmark for explanation quality
- Explanations: 1–3 sentences, in the lesson's vocabulary, teaching the concept rather than restating the question

## Common distractor patterns (watch for in existing questions)

- Questions that test generic ML facts instead of this lesson's specific engineering pattern
- Explanations that just repeat the correct option verbatim without adding pedagogical value

## Do not

- Change any question text, options, or `correct` index during `fill_explanations`.
- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
