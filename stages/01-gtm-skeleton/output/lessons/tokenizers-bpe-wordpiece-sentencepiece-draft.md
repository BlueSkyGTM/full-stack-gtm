# Tokenizers: BPE, WordPiece, SentencePiece

## Hook
Why "GTM" is one token in GPT-4 but three tokens in LLaMA — and why that difference changes your API bill, your prompt design, and your output quality. Every LLM you call in a workflow first shreds your input through a tokenizer. If you don't know the shredder's pattern, you can't predict the output.

## Concept
Subword tokenization as a compression scheme: models don't read words, they read integer indices into a fixed vocabulary. The three dominant algorithms (BPE, WordPiece, SentencePiece) solve the same problem — representing any text with a finite vocab — using different merge selection criteria. The choice of algorithm determines how names, jargon, URLs, and multilingual text get chunked.

## Mechanism
**BPE** (GPT-2, GPT-4, Claude): Initialize vocabulary as all individual characters. Count all adjacent symbol pairs across the corpus. Merge the most frequent pair into a new symbol. Repeat until vocabulary reaches target size. Result: common words stay whole; rare words split into frequent subwords.

**WordPiece** (BERT, DistilBERT): Same initialization, different selection. Instead of raw frequency, score each candidate merge by the likelihood it would increase given the current model. Select the merge that maximizes language model score. Result: merges prioritize linguistic coherence over raw frequency. Prefixed with `##` for non-initial subwords.

**SentencePiece** (T5, LLaMA, Mistral): Treats input as raw UTF-8 bytes. No pre-tokenization into words — whitespace is encoded as a special token (`▁`). Language-agnostic: same code path for English, Korean, code, or mixed content. BPE or Unigram algorithm applied at the byte level.

Walk through the same input string ("`B2B SaaS revenue: $2.3M`") tokenized by all three. Print vocab indices and merge traces to show where each algorithm splits differently.

## Use It
Token counts determine API cost per call in production GTM workflows. When Clay or any AI-powered enrichment tool processes a list of company names through an LLM, the tokenizer decides whether "Zuora" is one token or three. That changes your batch sizing, your rate limit calculations, and your per-record cost. [CITATION NEEDED — concept: tokenizer-aware prompt cost estimation in Clay workflows]. For Zone 1 (Foundations): tokenization literacy is prerequisite for prompt engineering. For Zone 2 (AI Workflows): tokenizer choice in your model selection dictates how firmographic data, technical jargon, and non-English company names survive the round trip.

## Ship It
- **Easy**: Tokenize 10 company names with three different tokenizer implementations. Print the token count and tokens for each. Identify which tokenizer handles them most efficiently.
- **Medium**: Implement BPE merge training from scratch on a corpus of 100 GTM-related sentences. Run the merge loop for 50 iterations. Print the vocabulary at intervals. Tokenize a held-out sentence using the trained vocabulary.
- **Hard**: Build a tokenizer cost estimator. Given a list of 1000 enrichment prompts and a target model's tokenizer, predict total token count, estimate API cost at current pricing, and flag inputs that exceed context windows. Compare predictions across BPE (GPT-4), WordPiece (BERT), and SentencePiece (LLaMA) tokenizers.

## Extend
- **Tokenizer bias**: BPE vocabularies trained on web text under-segment common English but over-segment names, code, and non-English text. Implications for GTM data where company names are often unusual strings.
- **Unigram tokenizer** (SentencePiece alternative): Instead of bottom-up merging, starts with a large vocabulary and prunes tokens that contribute least to likelihood. Used in T5 and some multilingual models.
- **Tokenizer is not the model**: Switching tokenizer requires retraining. You cannot swap SentencePiece into a BERT checkpoint.
- **Special tokens**: `[CLS]`, `[SEP]`, `<s>`, `</s>` — control tokens added during pre-training, not learned by the tokenizer algorithm. Missing them in your input causes silent degradation.
- **Further reading**: Sennrich et al. (2016) — original BPE paper for NMT; Schuster & Nakajima (2012) — Japanese/Korean segmentation that inspired SentencePiece; Kudo & Richardson (2018) — SentencePiece specification.