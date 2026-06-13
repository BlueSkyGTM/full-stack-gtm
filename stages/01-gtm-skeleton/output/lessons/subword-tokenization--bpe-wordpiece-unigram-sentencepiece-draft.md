# Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece

---

## Beat 1: Hook

Every LLM API bill is denominated in tokens — but a token is not a word, and it is not a character. It is an artifact of whichever subword algorithm the model's creators chose during training. If you cannot predict how your text tokenizes, you cannot predict cost, latency, or context window exhaustion. This lesson breaks the four algorithms that determine what your model actually sees.

---

## Beat 2: Concept

**Core mechanism:** Subword tokenization interpolates between word-level (large vocab, sparse) and character-level (small vocab, long sequences). The algorithm builds a vocabulary of variable-length units — common words stay whole, rare words split into recoverable fragments.

**Four algorithms, four strategies:**

- **BPE (Byte Pair Encoding):** Start with characters. Count all adjacent pairs in corpus. Merge the most frequent pair into a new symbol. Repeat until vocab size is reached. Greedy frequency heuristic. Used by GPT-2, GPT-4 (via tiktoken), Llama.
- **WordPiece:** Same merge loop as BPE, but instead of picking the most frequent pair, pick the pair that maximizes likelihood of the training data under a language model. Used by BERT, DistilBERT.
- **Unigram:** Start with a massive vocabulary. Score each subword by how much removing it increases corpus loss. Prune the bottom X%. Repeat until target vocab size. Probabilistic, not deterministic — each input has multiple valid tokenizations, selected by Viterbi at inference. Used by T5, ALBERT.
- **SentencePiece:** Not an algorithm — a library. Implements BPE and Unigram over raw byte streams, with no language-specific pre-tokenization (no `.split()` on spaces). The space character becomes a regular token (`▁`). Makes tokenization language-agnostic.

**Why it matters for practitioners:** Tokenizer choice is locked at model training time. You don't pick your tokenizer at inference — you inherit it. But you need to count tokens accurately to estimate cost and fit context windows.

---

## Beat 3: Demonstration

Code examples showing each algorithm in action:

1. **BPE from scratch** — implement the merge loop on a small corpus, print vocab at each step, show merge order.
2. **HuggingFace tokenizers** — load `bert-base-uncased` (WordPiece), `gpt2` (BPE), `t5-small` (Unigram) and tokenize the same sentence, print token IDs and subword pieces side-by-side.
3. **SentencePiece** — train a Unigram model on a sample corpus, encode/decode text, print the vocabulary with scores.
4. **Token counting for cost** — take a real prompt, count tokens via each tokenizer, multiply by published API prices, print a cost comparison table.

All output is printed to terminal. No notebooks, no browsers.

---

## Beat 4: Guided Exercise

Three exercises, increasing difficulty:

- **Easy:** Given a small corpus and a list of pre-computed pair frequencies, execute three BPE merge steps by hand and print the resulting vocabulary.
- **Medium:** Load three HuggingFace tokenizers, tokenize a batch of 10 real-world email subject lines, print a table showing token count per tokenizer per subject line, and identify which tokenizer is most efficient for this domain.
- **Hard:** Train a SentencePiece Unigram model on a domain-specific corpus (e.g., scraped company descriptions), then compare its token efficiency on held-out domain text vs. GPT-2's BPE tokenizer. Print compression ratio and vocabulary overlap percentage.

---

## Beat 5: Use It

**GTM Redirect:** Token counting is the cost backbone of any LLM-powered GTM workflow — enrichment prompts, classification pipelines, personalized email generation. This is foundational for Zone 1 (AI Foundations). When you batch-process thousands of company profiles through an LLM for scoring or personalization, the tokenizer determines your actual spend. A model that tokenizes "SaaS" as one token vs. three can change your API bill by 30% at scale.

Specific application: before running a Clay waterfall enrichment on 10,000 accounts, estimate token cost by tokenizing your prompt template + a sample of company data through the target model's tokenizer. Print the projected cost. If it exceeds budget, truncate or simplify the prompt before you run the table.

---

## Beat 6: Ship It

**Project:** Build a CLI token-cost estimator.

- Input: a text file (or stdin) containing prompts — one per line.
- Flags: `--model` (gpt-4, gpt-3.5-turbo, claude-3-haiku, etc.), `--count` (number of rows to process).
- Behavior: load the correct tokenizer for the model, tokenize each line, compute total tokens, multiply by the model's per-1K-token price (hardcoded price table), print total estimated cost.
- Output format: `Model: gpt-4 | Tokens: 45,230 | Est. Cost: $1.36`

Ship it as a single Python script. No dependencies beyond `tiktoken` or `transformers`.

---

## Learning Objectives

1. Implement the BPE merge loop from scratch on a toy corpus and print the resulting vocabulary at each step.
2. Compare tokenization outputs and token counts across BPE, WordPiece, and Unigram for the same input text.
3. Train a SentencePiece model on raw text and configure it for both BPE and Unigram modes.
4. Calculate LLM API cost from token counts using model-specific tokenizers and published pricing.
5. Identify which tokenizer a model uses from its configuration files ( `tokenizer_config.json`, `tokenizer.model` ) and output patterns ( `##` prefix = WordPiece, `Ġ` prefix = BPE, `▁` prefix = SentencePiece).