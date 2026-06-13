# Multimodal Evaluation

## Hook It
Evaluating a text-only model is straightforward: exact match, BLEU, ROUGE, F1. Evaluating a model that reasons across text, images, audio, and video is harder because "correct" becomes subjective and modality-specific. This lesson covers the mechanisms for measuring whether multimodal outputs are actually good.

## Map It
Break down the evaluation landscape into: (1) modality-specific metrics adapted from single-mode tasks, (2) cross-modal alignment scores that measure whether text and image actually correspond, and (3) judge-based evaluation using LLMs as automated raters. Explain the taxonomy: generative tasks (image captioning, text-to-image) vs. retrieval tasks (text-image retrieval) vs. reasoning tasks (visual QA). Name specific metrics (CLIPScore, BLEU, CIDEr, retrieval recall@k) but only after explaining what problem each solves.

## Build It
Write a working evaluation script that:
- Computes CLIPScore between generated captions and reference images using `transformers` and `torch`
- Computes CIDEr for image captioning against reference captions
- Runs a side-by-side comparison of two multimodal outputs using an LLM-as-judge pattern with structured prompts
- Prints all scores to stdout with labeled output

Exercise hooks:
- **Easy:** Run the provided CLIPScore evaluation on a set of image-caption pairs; identify which pairs score below 0.7 and hypothesize why.
- **Medium:** Implement a retrieval recall@k metric for a text-to-image retrieval task; compare against random baseline.
- **Hard:** Build a multi-metric evaluation pipeline that combines CLIPScore, CIDEr, and LLM-judge into a single weighted score; tune weights on a small labeled dataset and report correlation with human judgments.

## Use It
**GTM Redirect:** This maps to the enrichment and AI scoring patterns in GTM workflows where you're evaluating multimodal content — logo detection on company websites, banner ad analysis, or sales collateral that combines text and visuals.

The specific mechanism: when your enrichment pipeline uses a vision-language model to extract signals from prospect websites (e.g., "does this company's homepage show enterprise pricing?"), you need to measure whether those extractions are accurate before piping them into a scoring waterfall in Clay or similar. Multimodal evaluation metrics tell you whether to trust the signal.

If the AI concept doesn't map cleanly to a specific GTM workflow, the redirect is: **foundational for Zone 2 (Enrichment) — evaluating multimodal enrichment quality.**

[CITATION NEEDED — concept: Clay waterfall integration with multimodal enrichment signals]

## Ship It
Production considerations for multimodal evaluation:
- Cost tradeoffs: CLIPScore is cheap (one forward pass), LLM-as-judge is expensive (per-token API calls). When to use which.
- Latency: automated evaluation pipelines that run on every model update vs. periodic batch evaluation
- Human-in-the-loop: sampling strategy for human evaluation to validate automated metrics
- Metric drift: monitoring whether your evaluation metrics correlate with actual downstream GTM performance over time

Exercise hooks:
- **Easy:** Instrument a CLIPScore evaluation with timing and cost tracking; report the price-per-evaluation.
- **Medium:** Design an A/B evaluation framework that compares two vision-language models on the same enrichment task using multiple metrics; output a recommendation.
- **Hard:** Implement a monitoring script that tracks evaluation metric drift over time and alerts when CLIPScore drops below a threshold on a held-out test set.

## Extend It
- VQA benchmarks (VQAv2, GQA, TextVQA) and what they test
- Multimodal LLM benchmarks (MMMU, MMMU-pro, MathVista) and their limitations
- The subjective evaluation problem: why human agreement on multimodal quality is low (inter-annotator agreement metrics)
- Emerging approaches: reward models for multimodal alignment
- The gap between academic benchmarks and production GTM use cases

No exercise hooks in Extend It. Just references and reading.