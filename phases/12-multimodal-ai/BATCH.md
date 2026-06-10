# Phase 12 — Multimodal AI (quiz factory)

## Focus

Models that bridge two or more modalities: vision-language (CLIP, BLIP, LLaVA, GPT-4V), audio-language, video understanding, multimodal generation, cross-modal retrieval, and the embedding/alignment strategies that connect modalities.

## Scrape hints

- `docs/en.md`: contrastive training objectives (InfoNCE / CLIP loss), cross-attention vs concatenation fusion, modality tokenisation strategies, evaluation benchmarks (VQA, COCO captions, MMBench)
- `code/main.py`: often demonstrates embedding extraction, similarity scoring, or simple VQA pipeline
- Vocabulary: `glossary/terms.md` for contrastive loss, image patch tokenisation, cross-modal alignment, vision encoder, multimodal fusion

## Style anchor

- No gold quiz in this phase yet — use `phases/08-generative-ai/19-visual-autoregressive-var/quiz.json` for format reference
- pre = the modality gap problem, check = alignment/fusion mechanism, post = code demo or production deployment

## Common distractor patterns

- Confuse CLIP (contrastive, no generation) with BLIP/BLIP-2 (captioning + contrastive)
- Confuse cross-attention fusion (Q from one modality, K/V from other) with concatenation followed by self-attention
- Conflate image patch tokenisation (ViT-style) with VQ tokenisation (discrete visual tokens)
- Mix zero-shot CLIP with few-shot LLaVA (CLIP has no language decoder)
- Confuse InfoNCE temperature with softmax temperature in generation

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
