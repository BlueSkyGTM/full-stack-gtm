# Machine Translation

## Hook It
The transformer architecture was invented to solve translation. Every LLM you deploy for GTM descends from this specific problem: mapping a sequence in one language to a sequence in another while preserving meaning.

## Ground It
Covers the encoder-decoder pattern, attention as a solution to information bottleneck, beam search versus greedy decoding, and evaluation via BLEU. Explains why translation is harder than classification — the output length is unknown, word alignment is many-to-many, and fluency requires modeling entire sequences, not isolated tokens.

## Show It
Build a working translator using an open-source sequence-to-sequence model. Compare greedy versus beam decoding on the same input. Compute BLEU score against a reference translation to make quality observable. All code runs in terminal with printed output showing source text, translated text, and score.

## Use It
Translation applies directly to multilingual outbound and content localization — the same mechanisms that power machine translation underpin the "localize this landing page" and "translate this email sequence" operations in GTM stacks. Redirect: **Zone 2 (Content Engine)** — specifically, using translation to generate region-specific variants of messaging without hiring native speakers for every market. [CITATION NEEDED — concept: GTM content localization workflow using MT APIs]

## Ship It
Covers latency budgets for real-time translation in chat versus batch translation for email campaigns. Addresses language detection as a prerequisite step, handling code-switching (mixed-language input), and the cost curve of API-based translation versus self-hosted models. Includes a decision framework: when BLEU threshold justifies human review.

## Prove It
- **Easy:** Translate a batch of sentences and print BLEU scores per sentence.
- **Medium:** Implement a function that detects source language, translates to target, and flags low-confidence outputs below a BLEU threshold.
- **Hard:** Build a comparison script that runs the same input through two different translation approaches (e.g., different models or decoding strategies), computes BLEU for both against references, and outputs a ranked quality report.