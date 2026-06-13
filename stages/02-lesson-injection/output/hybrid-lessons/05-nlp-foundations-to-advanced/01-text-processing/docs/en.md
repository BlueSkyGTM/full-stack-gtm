# Text Processing — Tokenization, Stemming, Lemmatization

> Language is continuous. Models are discrete. Preprocessing is the bridge.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Learning Objectives

1. Implement word-level and regex-based tokenizers that split raw text into processable units, and compare their output distributions
2. Apply Porter and Snowball stemmers to the same corpus and identify cases where each produces non-word stems
3. Configure a spaCy lemmatization pipeline that uses part-of-speech context to resolve ambiguous forms
4. Evaluate which normalization method is appropriate for a given text classification or matching task, using vocabulary size and semantic preservation as criteria

---

## The Problem

A model cannot read "The cats were running." It reads integers. Before any classifier, any embedding model, any rules engine can touch text, that text must be split into units and those units must be reduced to a canonical form. Every NLP system opens with the same three questions: where does a word start, what is the root of the word, and how do we treat "run", "running", and "ran" as the same thing when it helps — and as different things when it doesn't.

Get tokenization wrong and the model learns from garbage. If your tokenizer treats `don't` as one token but `do n't` as two, the training distribution splits. If your stemmer collapses `organization` and `organ` to the same stem, topic modeling produces noise. If your lemmatizer needs part-of-speech context but you don't pass it, verbs get treated as nouns — `saw` (past tense of "see") and `saw` (the tool) collapse to the same token.

The practical cost is silent degradation. Your support ticket classifier drops 3% accuracy. Your ICP keyword matching misses prospects who use "analysing" instead of "analyzing." Your clustering puts "AI consultancy" and "AI consultants" in different buckets. None of these throw errors. They just make your outputs worse in ways that are hard to trace back to preprocessing.

This lesson builds the three operations from scratch — a regex tokenizer, a rule-based stemmer, a dictionary-based lemmatizer — then shows how NLTK and spaCy implement the same mechanisms so you can see the tradeoffs and choose the right one for your task.

---

## The Concept

Three operations sit at the entrance of every NLP pipeline. Each has a specific job and a specific failure mode.

**Tokenization** scans a string and splits it into substrings called tokens. The algorithm is delimiter-based: word-level tokenization splits on whitespace and punctuation, subword tokenization (BPE, WordPiece) splits frequent words whole and rare words into learned fragments, and character-level tokenization treats each character as a unit. The choice of tokenizer determines your vocabulary space — the set of unique tokens the model will ever see. A word-level tokenizer on a 10-million-word corpus might produce 50,000 unique tokens. A BPE tokenizer on the same corpus produces 30,000 subword tokens that can combine to represent any word, including unseen ones. For classical NLP (Naive Bayes, TF-IDF), word-level is standard. For transformers, subword is standard.

**Stemming** applies rule-based truncation. The Porter stemmer, the most common algorithm, iterates over five steps of suffix-stripping rules: it removes plurals (`cats → cat`), then past tense (`running → run`), then progressive forms, then common derivational suffixes (`organization → organiz`). No dictionary lookup. No part-of-speech reasoning. Pure pattern matching. This makes it fast — microseconds per word — and deterministic. It also makes it wrong in predictable ways: `organization → organ` (collapses two unrelated words), `better → bet` (ignores comparative form), `relational → relat` (produces a non-word). About 15-20% of stems are either non-words or incorrect roots. Stemming optimizes for recall: variants collapse to the same stem, so search and fuzzy matching benefit.

**Lemmatization** maps each word to its dictionary root form (lemma) using a morphological lexicon. The algorithm looks up the word in a pre-built table that encodes irregular forms (`ran → run`, `better → good`, `mice → mouse`) and regular inflections (`cats → cat`, `running → run`). Because the same surface form can be different parts of speech — `saw` is both a past-tense verb (lemma: `see`) and a noun (lemma: `saw`) — a lemmatizer needs part-of-speech context to disambiguate. This makes it slower than stemming (dictionary lookup + POS tagging) but accurate. Every output is a valid word.

```mermaid
flowchart LR
    A["Raw text:\n'The cats were running'"'] --> B["Tokenizer"]
    B --> C["['The', 'cats', 'were', 'running']"]
    C --> D{Normalization method}
    D -->|"Stemmer\n(suffix rules)"| E["['the', 'cat', 'were', 'run']"]
    D -->|"Lemmatizer\n(dict + POS)"| F["['the', 'cat', 'be', 'run']"]
    E --> G["Vocabulary: ~45k\nNon-words: ~15%"]
    F --> H["Vocabulary: ~35k\nNon-words: 0%"]
    G --> I["Best for:\nfuzzy search, recall"]
    H --> J["Best for:\nclassification,\nsemantic matching"]
```

The tradeoff is between speed and accuracy, between recall and precision. Stemming collapses more aggressively — `analysis`, `analyze`, `analysing` all become `analy` — which is ideal when you want broad recall in a search index or intent-signal detector. Lemmatization preserves meaning — `analysis` stays `analysis`, `analyze` becomes `analyze` — which is ideal when you extract features for a classifier that needs to distinguish "data analysis services" from "analytical chemistry." In practice, many pipelines stem for search and lemmatize for classification.

---

## Build It

Three runnable examples. Each builds a mechanism from first principles, then shows the library equivalent.

### Example 1: Regex Word Tokenizer — Built from Scratch vs NLTK

The simplest tokenizer splits on non-word characters. `re.findall(r'\w+', text)` grabs every maximal sequence of letters, digits, and underscores. This handles most cases but breaks on contractions (`don't` → `don`, `t`) and hyphens (`state-of-the-art` → four tokens). NLTK's `word_tokenize` handles these by applying a set of punctuation rules learned from the Penn Treebank corpus.

```python
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab', quiet=True)

text = "We're building state-of-the-art NLP pipelines. Don't overthink it."

raw_tokens = text.split()
regex_tokens = re.findall(r'\w+', text)
nltk_tokens = word_tokenize(text)

print("Raw split (whitespace):")
print(raw_tokens)
print(f"Count: {len(raw_tokens)}\n")

print("Regex \\w+ :")
print(regex_tokens)
print(f"Count: {len(regex_tokens)}\n")

print("NLTK word_tokenize:")
print(nltk_tokens)
print(f"Count: {len(nltk_tokens)}\n")

vocab_raw = len(set(t.lower() for t in raw_tokens))
vocab_nltk = len(set(t.lower() for t in nltk_tokens))
print(f"Unique tokens (raw): {vocab_raw}")
print(f"Unique tokens (nltk): {vocab_nltk}")
```

Run this and you will see three different token counts from the same input string. The whitespace split keeps punctuation attached to words (`pipelines.`), the regex split drops the apostrophe in `don't`, and NLTK separates punctuation as standalone tokens while keeping `n't` as a unit. The vocabulary sizes diverge accordingly. This is the first decision in any NLP pipeline: what counts as a token.

### Example 2: Stemming — Porter vs Snowball

The Porter stemmer applies a fixed sequence of suffix-stripping rules. The Snowball stemmer (also by Martin Porter, a later revision) applies a stricter rule set with minor improvements for English. Both are deterministic — same input, same output, every time.

```python
from nltk.stem import PorterStemmer, SnowballStemmer

words = [
    "running", "runs", "ran",
    "better", "best", "good",
    "organization", "organ", "organize",
    "relational", "relationship", "relating",
    "analyzing", "analyzed", "analysis",
    "happily", "happiness", "happy"
]

porter = PorterStemmer()
snowball = SnowballStemmer("english")

print(f"{'WORD':<18}{'PORTER':<16}{'SNOWBALL':<16}{'NOTES'}")
print("-" * 66)

for w in words:
    p = porter.stem(w)
    s = snowball.stem(w)
    note = ""
    if not p.isalpha() or len(p) < 3:
        note = "<-- non-word stem"
    if p != s:
        note += " [diverge]"
    print(f"{w:<18}{p:<16}{s:<16}{note}")

org_collapse = porter.stem("organization")
organ_stem = porter.stem("organ")
print(f"\norganization -> {org_collapse}")
print(f"organ        -> {organ_stem}")
print(f"Collapsed: {org_collapse == organ_stem}")
```

Run this and observe: `organization` and `organ` produce the same stem. `better` becomes `bet`. `analysis` becomes `analy`. These are not bugs in the algorithm — they are the expected output of rule-based truncation without a dictionary. The Snowball stemmer produces identical output in most cases but diverges on a few words where Porter's rules over-truncate.

### Example 3: Lemmatization with POS Context

The WordNet lemmatizer looks up each word in the WordNet morphological database. Without a part-of-speech tag, it defaults to noun — which means `running` (a verb form) gets treated as a noun and stays `running`. With the correct POS tag, it becomes `run`.

```python
import spacy
nlp = spacy.load("en_core_web_sm")

sentences = [
    "The cats were running faster than the dogs.",
    "She saw the problem and organized a team.",
    "This is better than the best alternative.",
    "Our organization provides data analysis services.",
]

print("=== spaCy Lemmatization (with automatic POS) ===\n")

for sent in sentences:
    doc = nlp(sent)
    pairs = [(token.text, token.lemma_) for token in doc]
    changed = [(t, l) for t, l in pairs if t.lower() != l.lower()]
    print(f"Text:     {sent}")
    print(f"Lemmas:   {' '.join(token.lemma_ for token in doc)}")
    print(f"Changed:  {changed}")
    print()

text = "The organizations were analyzing relationships between customers."
doc = nlp(text)

stems = [porter.stem(t.text.lower()) for t in doc]
lemmas = [t.lemma_ for t in doc]

print(f"{'TOKEN':<18}{'STEM':<16}{'LEMMA':<16}{'MATCH?'}")
print("-" * 50)
for token, stem, lemma in zip(doc, stems, lemmas):
    match = "yes" if stem == lemma else "NO"
    print(f"{token.text:<18}{stem:<16}{lemma:<16}{match}")
```

Run this and observe the critical difference: `saw` (the verb) lemmatizes to `see`, while a stemmer would produce `saw`. `better` lemmatizes to `good`. `organizations` lemmatizes to `organization` (correct root) while stemming produces `organ` (wrong root). The POS tagger inside spaCy runs automatically — it reads the sentence context and assigns a tag before lemmatization, which is why `running` in "were running" correctly becomes `run` rather than staying as `running`.

---

## Use It

The GTM application of tokenization and normalization maps to Zone 1 (ICP Enrichment) and Zone 3 (Outbound Personalization). When you scrape a company's website for enrichment, the raw description text is noisy — mixed cases, punctuation, industry jargon variants. The normalization pipeline you just built turns "AI-powered marketing automation platform" into a token sequence that can be matched against a taxonomy entry for "marketing automation." Without lemmatization, `analysing` and `analyzing` and `analyzed` are three different features. With stemming, they collapse to `analy` — one feature, broader recall. With lemmatization, they collapse to `analyze` or `analysis` — one canonical form, preserving the distinction between the verb and the noun.

For intent-signal detection — scraping job postings, news articles, or press releases for signs that a company is in your ICP — stemming gives you higher recall. If your trigger keyword is "scaling," you also want to match "scaled," "scales," and "scale-up." A stemmer collapses all of these to `scal`. The cost is false positives: "scale" (the weighing device) and "scale" (the musical concept) also match. For intent detection, that tradeoff is usually acceptable — you are fishing for signal, not building a precise classifier.

For classification — scoring whether a company is a good fit for your product based on their description text — lemmatization preserves more signal. A TF-IDF vectorizer built on lemmatized tokens produces cleaner features because the vocabulary is smaller (no duplicate word forms) and every token is a real word (no garbage stems that the classifier has to learn to ignore). The classifier can distinguish "consultancy" (services business) from "consultant" (individual) because the lemmatizer keeps them distinct, while a stemmer would collapse both to `consult`.

In Clay, this pattern appears as a formula field that lowercases and strips punctuation before matching against a lookup table. The underlying mechanism is identical to what you built above — tokenize, normalize, match. The difference is that Clay's formula language handles the plumbing, while Python gives you control over which normalizer you use and why.

For outbound personalization (Zone 3), the same normalization pipeline preprocesses prospect emails or LinkedIn activity before extracting topics for personalization. If you are clustering prospects by what they talk about, stemming or lemmatization reduces the feature space and prevents "marketing," "marketed," and "marketer" from becoming three separate clusters. The choice depends on whether your downstream step is a fuzzy keyword match (stem) or a semantic classifier (lemmatize).

---

## Ship It

To ship a normalization pipeline into a production GTM workflow — say, enriching incoming company records before ICP classification — you need three things: a single function that chains tokenization → normalization → output, a way to measure its impact, and a fallback when the input is empty or non-text.

The pipeline below takes a raw company description and returns either a list of stems (for fuzzy matching) or a list of lemmas (for classification). It handles empty strings, non-string inputs, and very long texts by capping at 10,000 characters to avoid spaCy processing timeouts on pathological inputs.

```python
import re
import spacy
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
porter = PorterStemmer()

def normalize_for_matching(text, method="lemma", max_chars=10000):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    text = text[:max_chars].lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    if method == "stem":
        tokens = text.split()
        return [porter.stem(t) for t in tokens if len(t) > 1]

    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 1]

descriptions = [
    "AI-powered marketing automation platform for B2B SaaS companies.",
    "We provide data analysis and business intelligence consulting services.",
    "",
    None,
    "Scaling our engineering team. Hiring senior developers and PMs.",
]

print("=== ICP Enrichment Pipeline ===\n")
for i, desc in enumerate(descriptions):
    stems = normalize_for_matching(desc, method="stem")
    lemmas = normalize_for_matching(desc, method="lemma")
    print(f"Record {i}: {(desc or 'EMPTY')[:60]}...")
    print(f"  Stems ({len(stems)}): {stems[:8]}")
    print(f"  Lemmas ({len(lemmas)}): {lemmas[:8]}")
    print()

corpus = [d for d in descriptions if isinstance(d, str) and len(d) > 0]
all_words = set()
all_stems = set()
all_lemmas = set()

for desc in corpus:
    words = re.findall(r'\w+', desc.lower())
    all_words.update(words)
    all_stems.update(normalize_for_matching(desc, "stem"))
    all_lemmas.update(normalize_for_matching(desc, "lemma"))

print(f"Vocabulary (raw tokens):   {len(all_words)}")
print(f"Vocabulary (stemmed):      {len(all_stems)}")
print(f"Vocabulary (lemmatized):   {len(all_lemmas)}")
print(f"Reduction (stem):          {(1 - len(all_stems)/len(all_words))*100:.1f}%")
print(f"Reduction (lemma):         {(1 - len(all_lemmas)/len(all_words))*100:.1f}%")
```

The vocabulary reduction numbers at the bottom are the shipping metric. If your raw corpus has 50,000 unique tokens and stemming reduces that to 35,000 while lemmatization reduces it to 38,000, you know exactly what each method costs and gains. The stemmer reduces more (aggressive truncation collapses unrelated words) while the lemmatizer reduces less (preserves real distinctions) but produces valid words only. For a downstream classifier, this means fewer features to train on and less sparsity per feature.

The `disable=["ner", "parser"]` flag on spaCy is a production optimization — it turns off named entity recognition and dependency parsing, which the lemmatizer does not need. This cuts processing time roughly in half. On a batch of 10,000 company descriptions, this is the difference between a pipeline that runs in 90 seconds and one that runs in 180 seconds.

---

## Exercises

**Exercise 1: Tokenizer comparison.** Take the string `"I can't recommend this highly enough — 10/10 would buy again."` and run it through (a) `.split()`, (b) `re.findall(r'\w+', text)`, and (c) NLTK `word_tokenize`. Count the tokens in each. Which tokenizer preserves the contraction `can't`? Which one treats `10/10` as a single token? Write a function that returns all three results as a dictionary.

**Exercise 2: Find the collapse.** Using the Porter stemmer, find five pairs of English words that are unrelated in meaning but collapse to the same stem. For each pair, explain which suffix rule caused the collapse. Hint: try words ending in `-ation`, `-ational`, `-er`, and `-ly`.

**Exercise 3: POS-dependent lemmatization.** Run these two sentences through spaCy: `"I saw a bird"` and `"I cut the board with a saw."` Extract the lemma of `saw` in each case. Write a function that takes a word and returns a list of possible lemmas depending on POS context, using `token.pos_` to branch.

**Exercise 4: Pipeline for ICP matching.** Build a function `match_icp(company_description, icp_keywords)` that normalizes both the description and the keywords using the same method (stem or lemma), then returns the number of keyword matches. Test it against three company descriptions where the keyword appears in different inflected forms (e.g., keyword `"analyze"` should match `"analysing"`, `"analyzed"`, `"analysis"`). Compare match counts for stem vs lemma.

---

## Key Terms

**Token** — A substring produced by a tokenizer. Can be a word, a subword fragment, a punctuation mark, or a single character depending on the tokenization scheme.

**Vocabulary** — The set of unique tokens in a corpus. Tokenization and normalization choices directly determine vocabulary size, which affects model training time, memory, and sparsity.

**Stem** — The output of a rule-based suffix-stripping algorithm. Not guaranteed to be a real word. Produced by algorithms like the Porter stemmer and Snowball stemmer without dictionary lookup.

**Lemma** — The dictionary root form of a word. Always a valid word. Produced by morphological analysis that considers part of speech (e.g., `better → good`, `running → run`).

**Part of Speech (POS)** — A grammatical category assigned to each token (noun, verb, adjective, etc.). Lemmatizers require POS to disambiguate words with multiple possible root forms.

**WordNet** — A lexical database of English that maps words to their synsets (groups of synonymous words) and morphological root forms. Used by NLTK's lemmatizer for dictionary lookup.

**Subword tokenization** — A tokenization scheme (BPE, WordPiece, SentencePiece) that splits rare words into common fragments. Used by transformer models to handle out-of-vocabulary words without an `<UNK>` token.

---

## Sources

- Porter stemmer algorithm: Porter, M.F. (1980). "An algorithm for suffix stripping." *Program* 14(3), 130–137.
- Snowball stemmer revision: Porter, M.F. (2001). "Snowball: A language for stemming algorithms." Open-source implementation at snowballstem.org.
- WordNet morphological database: Miller, G.A. (1995). "WordNet: A Lexical Database for English." *Communications of the ACM* 38(11), 39–41.
- Penn Treebank tokenization standard used by NLTK `word_tokenize`: Marcus, M. et al. (1993). "Building a Large Annotated Corpus of English: The Penn Treebank." *Computational Linguistics* 19(2).
- spaCy pipeline architecture (pipeline component disabling for performance): [CITATION NEEDED — concept: spaCy pipeline component disabling performance benchmark]
- Zone 1 ICP enrichment normalization pattern: [CITATION NEEDED — concept: text normalization for ICP taxonomy matching in GTM workflows]
- Zone 3 intent-signal detection via stemming: [CITATION NEEDED — concept: stemming for fuzzy keyword recall in intent-signal detection]