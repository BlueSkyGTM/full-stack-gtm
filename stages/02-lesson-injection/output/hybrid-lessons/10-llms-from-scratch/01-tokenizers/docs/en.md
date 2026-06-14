# Tokenizers: BPE, WordPiece, SentencePiece

## Learning Objectives

- Implement BPE merge training from scratch and trace how vocabulary grows with each iteration
- Compare BPE, WordPiece, and SentencePiece tokenization on identical GTM-related input strings
- Compute token-level cost differences across tokenizer implementations for the same input text
- Diagnose tokenization failures on company names, URLs, and multilingual text using merge traces
- Configure tiktoken and sentencepiece libraries to inspect token IDs for production prompt strings

## The Problem

Every LLM you call in a GTM workflow first converts your text to integers. The tokenizer decides the mapping. "GTM" is one token in GPT-4's tokenizer but three tokens in LLaMA's. That difference propagates into your API bill, your prompt design, and your output quality. If you do not know the shredder's pattern, you cannot predict the output.

The gap between "Hello, world!" and `[15496, 11, 995, 0]` is the tokenizer. Every word, every space, every punctuation mark must be converted into an integer before the model can process it. This conversion is not neutral — it bakes assumptions into the pipeline that cannot be undone at inference time.

Get this wrong and your model wastes capacity encoding common words with multiple tokens. "unfortunately" becomes four tokens instead of one. Your 128K context window shrinks by 75% for text heavy in multi-syllable words. Get it right and the same window holds twice as much meaning. The difference between "this model handles code well" and "this model chokes on Python" often traces back to how the tokenizer was trained.

Every API call you make to GPT-4 or Claude is priced per token. Every token your model generates costs compute. The fewer tokens required to represent an output, the faster the end-to-end inference. Tokenization is not preprocessing — it is architecture.

## The Concept

Three naive approaches to text-to-integer conversion fail at scale. Character-level tokenization assigns each character a unique ID, keeping vocabulary tiny but ballooning sequence length — "B2B SaaS revenue" becomes 17 tokens instead of 4. Word-level tokenization assigns each unique word an ID, producing short sequences but exploding vocabulary size — every typo, every conjugation, every new company name ("Zuora," "Clari," "Gong") demands its own entry. A 500K vocabulary means a 500K-row embedding matrix consuming GPU memory for words seen once in training.

Subword tokenization splits the difference. Common words stay whole as single tokens. Rare words decompose into reusable subword fragments. "Zuora" might split into `["Zu", "ora"]` — not ideal, but both fragments appear elsewhere in the vocabulary and earn their keep. Three algorithms dominate: BPE, WordPiece, and SentencePiece. They solve the same problem — representing any text with a finite vocabulary — using different merge selection criteria.

**BPE (Byte Pair Encoding)** initializes the vocabulary as all individual characters. It counts adjacent symbol pairs across the training corpus, merges the most frequent pair into a new symbol, and repeats until the vocabulary reaches its target size. Used by GPT-2, GPT-4, and Claude. **WordPiece** uses the same character-level initialization but a different selection criterion: instead of raw frequency, it scores each candidate merge by the likelihood it would increase given the current model. It selects the merge that maximizes a language model score. Non-initial subwords are prefixed with `##`. Used by BERT and DistilBERT. **SentencePiece** treats input as raw UTF-8 bytes with no pre-tokenization into words. Whitespace is encoded as the special token `▁`. The same code path handles English, Korean, code, or mixed content. BPE or Unigram algorithm applied at the byte level. Used by T5, LLaMA, and Mistral.

```mermaid
flowchart TD
    A["Raw Input Text"] --> B{"Pre-tokenize into words?"}
    B -->|"Yes: BPE, WordPiece"| C["Split on whitespace/punctuation"]
    B -->|"No: SentencePiece"| D["Treat as raw UTF-8 bytes"]
    C --> E["Initialize vocab to individual characters"]
    D --> E
    E --> F{"Select merge candidate"}
    F -->|"BPE: Most frequent pair"| G["Merge pair, add to vocab"]
    F -->|"WordPiece: Max likelihood gain"| G
    F -->|"SentencePiece: BPE or Unigram on bytes"| G
    G --> H{"Vocab size reached?"}
    H -->|"No"| F
    H -->|"Yes"| I["Final Vocabulary"]
    I --> J["Tokenize new text by\napplying merges greedily"]
```

Walk through `"B2B SaaS revenue: $2.3M"` under each algorithm. BPE (GPT-4's tiktoken) produces something like `["B", "2", "B", " S", "aa", "S", " revenue", ":", " $", "2", ".", "3", "M"]` — 13 tokens. WordPiece (BERT) fragments differently because BERT's vocabulary was trained on Wikipedia, not SaaS jargon — "SaaS" is not a known word. SentencePiece (LLaMA) operates at the byte level and may chunk "revenue" differently depending on whether its training corpus included financial notation. There is no "correct" tokenization. There is only the tokenization your specific model was trained with, and you must inspect it to know how your input gets shredded.

## Build It

The clearest way to understand BPE is to implement the merge loop yourself. Here is a working implementation that trains on a GTM corpus and traces each merge decision.

```python
from collections import Counter

corpus = [
    "B2B SaaS revenue growth",
    "SaaS companies scale revenue",
    "B2B sales pipeline metrics",
    "revenue churn and retention",
    "SaaS ARR MRR and growth",
    "B2B GTM strategy and execution",
    "pipeline coverage and conversion",
    "customer acquisition cost",
    "lifetime value and churn",
    "SaaS revenue operations",
]

word_freqs = Counter()
for sentence in corpus:
    for word in sentence.split():
        word_freqs[word] += 1

splits = {word: list(word) for word in word_freqs}
vocab = set()
for word in splits:
    for char in word:
        vocab.add(char)

print(f"Initial vocabulary ({len(vocab)} chars): {sorted(vocab)}")
print()

def get_pair_stats(splits, word_freqs):
    pair_counts = Counter()
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts

def merge_pair(pair, splits):
    new_splits = {}
    for word, symbols in splits.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[word] = new_symbols
    return new_splits

num_merges = 15
for merge_idx in range(num_merges):
    pair_stats = get_pair_stats(splits, word_freqs)
    if not pair_stats:
        break
    best_pair = max(pair_stats, key=pair_stats.get)
    splits = merge_pair(best_pair, splits)
    new_token = best_pair[0] + best_pair[1]
    vocab.add(new_token)
    print(f"Merge {merge_idx + 1:2d}: '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'  (frequency: {pair_stats[best_pair]})")

print(f"\nFinal vocabulary size: {len(vocab)}")
```

When you run this, notice what BPE prioritizes. The pair "re" merges early because it appears across "revenue" and "retention." The algorithm has no linguistic knowledge — it is pure frequency counting. Now compare how production tokenizers handle the same GTM strings.

```python
import subprocess
import sys

def install_if_needed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install_if_needed("tiktoken")
install_if_needed("transformers")
install_if