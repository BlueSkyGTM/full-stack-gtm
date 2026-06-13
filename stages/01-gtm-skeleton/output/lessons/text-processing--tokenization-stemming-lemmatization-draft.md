# Text Processing — Tokenization, Stemming, Lemmatization

## Learning Objectives

1. Implement tokenization strategies that split raw text into processable units for downstream NLP tasks
2. Compare stemming versus lemmatization outputs on the same corpus and articulate the tradeoff
3. Configure a text normalization pipeline that reduces vocabulary size while preserving meaning
4. Evaluate which normalization method is appropriate for a given text classification or matching task

---

## Beat 1: Hook

Raw text is unstructured. Models and rules engines expect structured input. Before you can classify a support ticket, match a company description to an ICP, or cluster prospect emails, you need to break text apart and normalize it. This lesson covers the three operations that sit at the entrance of every NLP pipeline: tokenization, stemming, and lemmatization. Get these wrong and everything downstream degrades silently.

---

## Beat 2: Concept

**Tokenization** — The algorithm scans a string and splits it into substrings (tokens) based on delimiter rules. Word-level tokenization splits on whitespace and punctuation. Subword tokenization (BPE, WordPiece) splits frequent words whole and rare words into pieces. The choice of tokenizer determines your vocabulary space.

**Stemming** — A rule-based truncation algorithm strips suffixes using heuristic patterns (e.g., Porter stemmer: "running" → "run", "better" → "bet"). Fast, deterministic, no dictionary required. Produces non-words in ~15-20% of cases. Useful when recall matters more than precision.

**Lemmatization** — A dictionary-lookup algorithm maps each word to its root form (lemma) using a morphological lexicon (e.g., "better" → "good", "running" → "run"). Slower than stemming but produces valid words. Requires part-of-speech context for disambiguation ("saw" the verb vs. "saw" the noun).

**The tradeoff**: Stemming is fast and aggressive. Lemmatization is slower and accurate. For fuzzy matching and search, stemming is often sufficient. For feature extraction in classification pipelines, lemmatization preserves more signal.

---

## Beat 3: Use It

**GTM Redirect**: Zone 1 (ICP Enrichment) and Zone 3 (Outbound Personalization).

When you enrich a company record from a web scrape, the raw description is noisy. Tokenize it, lemmatize it, and you can match "AI-powered marketing automation platform" to "marketing automation" in your ICP taxonomy. Stemming gives you broader recall for fuzzy keyword matching in intent signals — "analyzing", "analyzed", "analysis" all collapse to the same stem. Lemmatization gives you cleaner features for a classifier that scores fit.

In Clay workflows, a formula field that normalizes text before matching against a lookup table implements this pattern manually. The underlying mechanism is identical.

---

## Beat 4: Code It

Three runnable examples. Each produces observable output.

### Example 1: Tokenization — Compare word-level and character-level splits

```python
import re

text = "Clay's GTM pipeline processes 1,200 leads/day."

word_tokens = re.findall(r'\b\w+\b', text.lower())
char_tokens = [c for c in text.lower() if c.isalnum()]

print("Word tokens:", word_tokens)
print("Count:", len(word_tokens))
print("Char tokens:", char_tokens)
print("Count:", len(char_tokens))
```

**Exercise hook (Easy)**: Run this tokenizer on a corpus of 5 company descriptions. Count the total vocabulary size before and after lowercasing.

### Example 2: Stemming vs Lemmatization — Side-by-side comparison

```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

words = ["running", "better", "studies", "geese", "corpora", "analyzed"]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print(f"{'Input':<15} {'Stem':<15} {'Lemma':<15}")
print("-" * 45)
for w in words:
    print(f"{w:<15} {stemmer.stem(w):<15} {lemmatizer.lemmatize(w):<15}")
```

**Exercise hook (Medium)**: Feed a paragraph of marketing copy through both stemmer and lemmatizer. Count how many tokens differ between the two outputs. Identify cases where the stemmer produces a non-word.

### Example 3: Full normalization pipeline — Vocabulary reduction measurement

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

corpus = [
    "The marketing team was running campaigns across multiple channels.",
    "She analyzed the analyses from three different studies.",
    "Better insights lead to better decisions and better outcomes."
]

lemmatizer = WordNetLemmatizer()

raw_vocab = set()
normalized_vocab = set()

for sentence in corpus:
    tokens = word_tokenize(sentence.lower())
    raw_vocab.update(tokens)
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    normalized_vocab.update(lemmas)

reduction = (1 - len(normalized_vocab) / len(raw_vocab)) * 100

print(f"Raw vocabulary size: {len(raw_vocab)}")
print(f"Lemmatized vocabulary size: {len(normalized_vocab)}")
print(f"Vocabulary reduction: {reduction:.1f}%")
print(f"\nTokens removed: {raw_vocab - normalized_vocab}")
```

**Exercise hook (Hard)**: Build a function that accepts a list of company descriptions and a normalization strategy (none, stem, lemma). Return a vocabulary overlap score between any two descriptions. Test whether stemming or lemmatization produces higher overlap for companies in the same industry vertical.

---

## Beat 5: Ship It

**Production considerations:**

- **Speed**: Stemming is ~10x faster than lemmatization. If you're normalizing millions of records in a batch pipeline, measure whether the accuracy gain from lemmatization justifies the runtime cost.
- **Consistency**: Always normalize before deduplication or matching. A lookup table that maps "SaaS" and "saas" and "Software-as-a-Service" to one canonical form is a hand-built lemmatizer. Document it.
- **Language**: Porter stemmer and WordNet lemmatizer are English-only. spaCy supports multilingual lemmatization out of the box. If your corpus includes non-English text, switch the library, not the approach.
- **GTM integration**: In a Clay waterfall, any formula that lowercases and strips punctuation before a lookup is doing tokenization + normalization. The mechanism is the same. [CITATION NEEDED — concept: Clay text normalization formula patterns]

**Deploy checklist:**
- [ ] Measure vocabulary size before and after normalization on a representative sample
- [ ] Confirm normalization does not collapse distinct meanings (e.g., "US" the country vs. "us" the pronoun)
- [ ] Log normalization time per 1K records
- [ ] Write a regression test: given a fixed input, assert the normalized output

---

## Beat 6: Quiz Hooks

These map directly to the learning objectives. Each is grounded in observable behavior from the code examples.

**Easy** (Objective 1):
- Given a string, predict the output of word-level tokenization vs. character-level tokenization. Count the tokens.
- Given two tokenization strategies, identify which produces a larger vocabulary on the same input.

**Medium** (Objective 2):
- Given the input word "better", predict the output of Porter stemming vs. lemmatization. Explain why they differ.
- Given a list of 10 words, classify each as: (a) stem and lemma match, (b) stem and lemma differ, (c) stemmer produces a non-word.

**Hard** (Objective 3 & 4):
- Given a corpus and a target vocabulary reduction threshold, configure a normalization pipeline (choose stem or lemma) that meets the threshold. Justify the choice.
- Given two GTM scenarios (fuzzy keyword matching for intent signals vs. feature extraction for a fit classifier), recommend stemming or lemmatization and explain the tradeoff in terms of precision and recall.