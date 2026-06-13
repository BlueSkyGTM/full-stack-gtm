# Multilingual NLP

## Hook

You've built a lead scoring model on English-language job titles and company descriptions. A surge of German, Japanese, and Portuguese prospects enters the pipeline. Your model's confidence collapses — not because the signal changed, but because your tokenizer was never designed to see those languages as anything other than noise. Multilingual NLP is the engineering discipline of making language-aware systems work across script boundaries, tokenization mismatches, and embedding space fragmentation.

## Concept

Covers the three failure modes that break monolingual systems at the language boundary: tokenization divergence (subword splits behave differently per language), embedding space misalignment (similar concepts land in different vector regions across languages), and script/encoding edge cases (Unicode normalization, right-to-left text, zero-width characters). Introduces the mechanism of shared subword vocabularies (SentencePiece, Byte-Pair Encoding) and how models like mBERT and XLM-R create a single embedding space across 100+ languages through joint training — and where that shared space breaks down (low-resource languages, code-switching).

## Demo

Working Python script that takes a list of company descriptions in five languages, runs language detection, tokenizes each with a multilingual tokenizer, and prints token counts and overlap statistics side by side — revealing where the same company description explodes in token count depending on language. Observable output: a table showing language, token count, and the first 10 subword tokens.

## Use It

GTM cluster: **Zone 3 — Enrichment / Qualification** (specifically, multilingual lead enrichment and international ICP matching). When enriching international prospects, monolingual NER and classification silently fail on non-English firmographic data. Mechanism: run multilingual NER (e.g., XLM-R-based) on company descriptions in their original language rather than translating first, because translation erases entity boundary signal. Build an enrichment function that detects language, routes to the appropriate processing path, and returns normalized firmographic fields regardless of input language. Redirect note: if your ICP is monolingual, this lesson is foundational for Zone 3 — the multilingual mechanism doesn't apply until you operate in multiple markets.

## Ship It

Deploy multilingual processing as a preprocessing stage in an enrichment pipeline. Covers the production decisions: when to translate-then-classify vs classify-in-original-language (latency vs accuracy tradeoff), how to handle language detection failures on short text (company names, job titles), and how to set confidence thresholds per language based on model training data distribution. Exercise hook: given a CSV of 200 international prospects with mixed-language descriptions, build a processing pipeline that outputs normalized firmographic data with a `language_confidence` field.

## Evaluate

Covers metrics that expose multilingual failure: per-language F1 scores (not aggregate), tokenization parity ratios (token count of non-English text / token count of English equivalent text), and cross-lingual embedding alignment scores using MRR or cosine similarity on translation pairs. Exercise hook: given model predictions on a multilingual test set, compute per-language classification accuracy and identify the three languages where the model degrades most — then hypothesize whether the cause is tokenization, training data volume, or script complexity.

---

**Exercise hooks only:**
- Easy: Detect language and print token count for 10 strings across 5 languages.
- Medium: Build a function that classifies company descriptions in 3 languages without translating, and compare accuracy against a translate-then-classify baseline.
- Hard: Given per-language accuracy data for XLM-R on 15 languages, diagnose whether accuracy degradation correlates with tokenization expansion ratio or training set size, and propose a targeted fix.