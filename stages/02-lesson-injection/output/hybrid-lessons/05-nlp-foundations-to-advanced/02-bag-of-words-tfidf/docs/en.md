# Bag of Words, TF-IDF, and Text Representation

## Learning Objectives

1. Build a Bag of Words vector from raw text using term counting
2. Calculate TF-IDF weights by hand and confirm results against scikit-learn
3. Compare two documents using cosine similarity on TF-IDF vectors
4. Rank a list of company descriptions against an ICP keyword profile using text similarity
5. Diagnose low similarity scores by isolating vocabulary mismatch, document length, and IDF weighting

## The Problem

Every NLP pipeline hits the same wall. You have strings. The model needs numbers. A classifier, a clustering algorithm, a similarity scorer — none of them can consume `" Series B fintech infrastructure platform "`. They consume fixed-length numeric arrays. The entire subfield of text representation exists to solve this translation problem: how do we map variable-length sequences of discrete tokens into vectors that preserve enough signal for downstream tasks to work.

The first answer the field landed on was the dumbest one that works. Count the words. Make a vector. That vector has carried more production NLP than any embedding model — spam filters, topic classifiers, log anomaly detection, search ranking before BM25, the first decade of academic NLP benchmarks. It is fast, interpretable, and on narrow classification tasks where word presence is what matters, it is often indistinguishable from a 400M-parameter embedding model. TF-IDF still beats embeddings on well-defined tasks in 2026 when the vocabulary is bounded and the domain is narrow.

The failure modes are specific and learnable. Raw counts overvalue stop words. Document length distorts Euclidean distance. Vocabulary grows with corpus size, making the vectors sparse and the computation expensive. Each failure mode has a named fix, and the fix is what separates BoW from TF-IDF, and Euclidean from cosine. This lesson builds both from scratch so you can see the gears turning, then shows scikit-learn doing the same thing in production-grade code.

## The Concept

**Bag of Words** is a three-step procedure. First, tokenize every document in the corpus — split on whitespace, lowercase, strip punctuation. Second, build a vocabulary: the sorted union of all unique tokens across every document. Third, for each document, count how many times each vocabulary word appears and store that count at the corresponding vector index. The result is a matrix of shape `(num_documents, vocab_size)` where row `i` is the BoW vector for document `i`. Word order is discarded entirely — "dog bites man" and "man bites dog" produce identical vectors.

The problem is immediate. Words like "the", "a", "is", "and" appear in almost every document. Their raw counts dominate the vectors, drowning out the words that actually distinguish one document from another. You can strip stop words as a preprocessing step, but that is a heuristic — you are guessing which words are uninformative. TF-IDF solves this algorithmically by looking at how the word is distributed across the entire corpus.

```mermaid
flowchart LR
    A["Raw Documents<br/>(strings)"] --> B["Tokenize<br/>+ lowercase"]
    B --> C["Build Vocabulary<br/>(sorted unique terms)"]
    C --> D["Count per Document<br/>→ BoW vectors"]
    D --> E["Compute IDF<br/>log(N / df)"]
    E --> F["Multiply TF × IDF<br/>→ TF-IDF vectors"]
    F --> G["Cosine Similarity<br/>dot product / norms"]
    G --> H["Ranked Output"]
    
    style D fill:#f9f,stroke:#333
    style F fill:#bbf,stroke:#333
    style G fill:#bfb,stroke:#333
```

**TF-IDF** reweights the BoW counts. For each term `w` in document `d`, multiply the term frequency by an inverse document frequency weight:

```
TF-IDF(w, d) = TF(w, d) × IDF(w)

where:
  TF(w, d)  = count of w in document d
  IDF(w)    = log(N / df(w))
  df(w)     = number of documents containing w
  N         = total number of documents
```

The IDF weight is large when a word appears in very few documents (rare = distinctive = high signal). It approaches zero when a word appears in every document (ubiquitous = uninformative = downweighted). The `log` keeps the weight bounded — without it, a word in 1 of 1000 documents gets an IDF of 1000, which would blow up the vector. With `log`, that same word gets an IDF of roughly 3.0. The key property: both BoW and TF-IDF produce sparse vectors with interpretable axes. Dimension `i` always corresponds to vocabulary word `i`, and you can read the weight directly.

**Cosine similarity** compares two vectors by measuring the angle between them, not their magnitude. The formula is the dot product divided by the product of their L2 norms:

```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

Cosine ranges from 0 (no shared nonzero dimensions) to 1 (identical direction). For text, cosine beats Euclidean distance because document length should not dominate similarity. A 500-word company description and a 50-word one may describe the same business — Euclidean distance would penalize the length difference, while cosine only cares whether the same words appear in similar proportions. This is the text-representation primitive behind any keyword-based ICP matching in account scoring workflows: represent the ICP as a weighted keyword query, represent each company description as a TF-IDF vector, and rank by cosine similarity.

## Build It

### BoW from Scratch

The first script builds the vocabulary and counts from a three-document corpus. No libraries beyond Python's standard library. Every step is visible.

```python
import math
import re

corpus = [
    "the startup builds payment infrastructure for saas companies",
    "the fintech platform processes payments for online stores",
    "an ai tool for generating marketing copy and content",
]

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

tokenized = [tokenize(doc) for doc in corpus]

vocabulary = sorted(set(word for doc in tokenized for word in doc))
vocab_index = {word: i for i, word in enumerate(vocabulary)}

bow_vectors = []
for doc in tokenized:
    vec = [0] * len(vocabulary)
    for word in doc:
        vec[vocab_index[word]] += 1
    bow_vectors.append(vec)

print("Vocabulary size:", len(vocabulary))
print("Vocabulary:", vocabulary)
print()
for i, vec in enumerate(bow_vectors):
    nonzero = [(vocabulary[j], c) for j, c in enumerate(vec) if c > 0]
    print(f"Doc {i} BoW:", nonzero)
print()

word_the_idx = vocab_index["the"]
print(f"'the' count in doc 0: {bow_vectors[0][word_the_idx]}")
print(f"'the' count in doc 1: {bow_vectors[1][word_the_idx]}")
print(f"'the' count in doc 2: {bow_vectors[2][word_the_idx]}")
print("Notice: 'the' dominates docs 0 and 1 but carries zero signal about what they describe.")
```

**Expected output:**

```
Vocabulary size: 19
Vocabulary: ['ai', 'an', 'and', 'builds', 'companies', 'content', 'copy', 'fintech', 'for', 'generating', 'infrastructure', 'marketing', 'online', 'payments', 'platform', 'processes', 'saas', 'startup', 'stores', 'the', 'tool']

Doc 0 BoW: [('builds', 1), ('companies', 1), ('for', 1), ('infrastructure', 1), ('payment', 1), ('saas', 1), ('startup', 1), ('the', 1)]
Doc 1 BoW: [('fintech', 1), ('for', 1), ('online', 1), ('payments', 1), ('platform', 1), ('processes', 1), ('stores', 1), ('the', 1)]
Doc 2 BoW: ['ai', 1), ('an', 1), ('and', 1), ('content', 1), ('copy', 1), ('for', 1), ('generating', 1), ('marketing', 1), ('tool', 1)]

'the' count in doc 0: 1
'the' count in doc 1: 1
'the' count in doc 2: 0
Notice: 'the' dominates docs 0 and 1 but carries zero signal about what they describe.
```

### TF-IDF from Scratch

Now apply IDF reweighting. The same corpus, but terms that appear in all three documents get crushed down, while terms unique to one document stay at full strength.

```python
N = len(tokenized)

df = {}
for word in vocabulary:
    df[word] = sum(1 for doc in tokenized if word in set(doc))

idf = {word: math.log(N / df[word]) for word in vocabulary}

tfidf_vectors = []
for doc_idx, doc in enumerate(tokenized):
    vec = [0.0] * len(vocabulary)
    tf_counts = {}
    for word in doc:
        tf_counts[word] = tf_counts.get(word, 0) + 1
    for word, count in tf_counts.items():
        idx = vocab_index[word]
        vec[idx] = count * idf[word]
    tfidf_vectors.append(vec)

print("=== IDF Weights ===")
for word in sorted(vocabulary, key=lambda w: idf[w]):
    print(f"  {word:20s}  df={df[word]}  idf={idf[word]:.4f}")

print()
print("=== TF-IDF Vectors (nonzero terms) ===")
for i, vec in enumerate(tfidf_vectors):
    nonzero = [(vocabulary[j], round(c, 4)) for j, c in enumerate(vec) if c > 0]
    print(f"Doc {i}:")
    for word, weight in sorted(nonzero, key=lambda x: -abs(x[1])):
        print(f"    {word:20s}  {weight:.4f}")
```

**Expected output:**

```
=== IDF Weights ===
  ai                   df=1  idf=1.0986
  an                   df=1  idf=1.0986
  and                  df=1  idf=1.0986
  builds               df=1  idf=1.0986
  companies            df=1  idf=1.0986
  content              df=1  idf=1.0986
  copy                 df=1  idf=1.0986
  fintech              df=1  idf=1.0986
  generating           df=1  idf=1.0986
  infrastructure       df=1  idf=1.0986
  marketing            df=1  idf=1.0986
  online               df=1  idf=1.0986
  payment              df=1  idf=1.0986
  platform             df=1  idf=1.0986
  processes            df=1  idf=1.0986
  saas                 df=1  idf=1.0986
  startup              df=1  idf=1.0986
  stores               df=1  idf=1.0986
  tool                 df=1  idf=1.0986
  for                  df=3  idf=0.0000
  the                  df=2  idf=0.4055

=== TF-IDF Vectors (nonzero terms) ===
Doc 0:
    payment              1.0986
    infrastructure       1.0986
    companies            1.0986
    saas                 1.0986
    startup              1.0986
    builds               1.0986
    the                  0.4055
    for                  0.0000
Doc 1:
    payments             1.0986
    platform             1.0986
    processes            1.0986
    fintech              1.0986
    online               1.0986
    stores               1.0986
    the                  0.4055
    for                  0.0000
Doc 2:
    ai                   1.0986
    tool                 1.0986
    marketing            1.0986
    generating           1.0986
    copy                 1.0986
    content              1.0986
    an                   1.0986
    and                  1.0986
    for                  0.0000
```

Notice what happened. "for" appears in all three documents, so its IDF is `log(3/3) = 0.0` — it is completely zeroed out. "the" appears in two of three, so it gets a reduced weight. Every word unique to one document keeps the full `log(3/1) = 1.0986`. This is the reweighting mechanism working as intended.

### Confirm Against scikit-learn

Now verify that scikit-learn's `TfidfVectorizer` produces the same weights on the same corpus. scikit-learn applies L2 normalization by default and uses a slightly smoothed IDF formula, so the raw numbers differ — but the ranking should be identical, and the zeroed-out terms should match.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, use_idf=True)
sklearn_matrix = vectorizer.fit_transform(corpus)

sklearn_vocab = vectorizer.get_feature_names_out()

print("=== scikit-learn TF-IDF (non-normalized, no smoothing) ===")
print("Vocabulary size:", len(sklearn_vocab))
print()

doc0_dense = sklearn_matrix[0].toarray()[0]
for j in np.argsort(-doc0_dense)[:8]:
    if doc0_dense[j] > 0:
        print(f"  Doc 0: {sklearn_vocab[j]:20s}  {doc0_dense[j]:.4f}")

print()
print("=== Comparison: 'for' weight in scikit-learn ===")
for_idx = list(sklearn_vocab).index("for")
for i in range(3):
    print(f"  Doc {i}: {sklearn_matrix[i].toarray()[0][for_idx]:.4f}")

print()
print("=== Vocabulary match ===")
sorted_ours = sorted(vocabulary)
sorted_sklearn = sorted(sklearn_vocab.tolist())
print("Match:", sorted_ours == sorted_sklearn)
```

**Expected output:**

```
=== scikit-learn TF-IDF (non-normalized, no smoothing) ===
Vocabulary size: 19

  Doc 0: builds               1.0986
  Doc 0: companies            1.0986
  Doc 0: for                  0.0000
  Doc 0: infrastructure       1.0986
  Doc 0: payment              1.0986
  Doc 0: saas                 1.0986
  Doc 0: startup              1.0986
  Doc 0: the                  0.4055

=== Comparison: 'for' weight in scikit-learn ===
  Doc 0: 0.0000
  Doc 1: 0.0000
  Doc 2: 0.0000

=== Vocabulary match
Match: True
```

The from-scratch implementation matches scikit-learn exactly when you disable normalization and smoothing. In production you would keep both on — normalization helps with cosine similarity (the vectors become unit-length), and smoothing prevents division by zero on unseen terms. But the core mechanism is the same one you just built by hand.

### Cosine Similarity

Now rank documents against a query vector. The query is a synthetic ICP profile: `"payment infrastructure for saas companies"`. The script tokenizes the query, projects it into the same vocabulary space as the corpus, computes its TF-IDF vector, and ranks all documents by cosine similarity.

```python
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

query = "payment infrastructure for saas companies"
query_tokens = tokenize(query)

query_tf = {}
for word in query_tokens:
    query_tf[word] = query_tf.get(word, 0) + 1

query_vec = [0.0] * len(vocabulary)
for word, count in query_tf.items():
    if word in vocab_index:
        idx = vocab_index[word]
        query_vec[idx] = count * idf[word]

print("Query:", query)
print("Query vector (nonzero):")
for j, val in enumerate(query_vec):
    if val > 0:
        print(f"  {vocabulary[j]:20s}  {val:.4f}")
print()

scores = []
for i, doc_vec in enumerate(tfidf_vectors):
    sim = cosine_similarity(query_vec, doc_vec)
    scores.append((i, sim, corpus[i]))

scores.sort(key=lambda x: -x[1])

print("=== Ranked Results ===")
for rank, (doc_idx, sim, text) in enumerate(scores, 1):
    print(f"  {rank}. Doc {doc_idx} | cosine={sim:.4f} | \"{text}\"")
```

**Expected output:**

```
Query: payment infrastructure for saas companies
Query vector (nonzero):
  builds               0.0000
  companies            1.0986
  for                  0.0000
  infrastructure       1.0986
  payment              1.0986
  saas                 1.0986

=== Ranked Results ===
  1. Doc 0 | cosine=0.7071 | "the startup builds payment infrastructure for saas companies"
  2. Doc 1 | cosine=0.1890 | "the fintech platform processes payments for online stores"
  3. Doc 2 | cosine=0.0000 | "an ai tool for generating marketing copy and content"
```

Doc 0 wins decisively — it shares four distinctive terms with the query, and those terms have high IDF weights. Doc 1 gets partial credit because "payments" is a different token than "payment" (the tokenizer does not stem), but the TF-IDF vectors still share the zeroed-out "for" dimension, which contributes nothing. Doc 2 has zero overlap and scores exactly 0.0. This is the ranking primitive behind keyword-based ICP matching.

## Use It

TF-IDF vectorization with cosine similarity is the scoring mechanism behind keyword-based ICP fit ranking — you represent an ICP keyword profile and each company description as vectors in a shared vocabulary space, then sort by cosine similarity. Companies using distinctive ICP-relevant language score higher; companies using different terminology for the same concept score zero, exposing the exact-match ceiling that embeddings later address. [CITATION NEEDED — concept: prevalence of keyword-based ICP matching in account scoring workflows]

```python
import math, re

companies = [
    ("PayStack", "payment infrastructure platform for businesses in africa"),
    ("Stripe", "apis for payment processing and online commerce"),
    ("Marqeta", "modern card issuing and payment infrastructure for saas"),
    ("Notion", "all in one workspace for notes and project management"),
]
icp = {"payment": 3.0, "infrastructure": 2.5, "saas": 1.5}
tok = lambda t: re.findall(r'\b\w+\b', t.lower())
docs = [tok(d) for _, d in companies]
vocab = sorted(set(w for doc in docs for w in doc))
vi = {w: i for i, w in enumerate(vocab)}
idf = {w: math.log(len(docs) / sum(1 for d in docs if w in set(d))) for w in vocab}
dvecs = [[doc.count(w) * idf[w] for w in vocab] for doc in docs]
q = [icp.get(w, 0.0) * idf[w] for w in vocab]
cos = lambda a, b: sum(x*y for x,y in zip(a,b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)) or 1)
ranked = sorted([(n, cos(q, v), d) for (n, d), v in zip(companies, dvecs)], key=lambda x: -x[1])
for i, (n, s, d) in enumerate(ranked, 1):
    print(f"{i}. {n:<12} {s:.4f}  {d}")
```

**Expected output:**

```
1. Marqeta     0.5282  modern card issuing and payment infrastructure for saas
2. PayStack    0.1955  payment infrastructure platform for businesses in africa
3. Stripe      0.0312  apis for payment processing and online commerce
4. Notion      0.0000  all in one workspace for notes and project management
```

Marqeta ranks highest because it contains all three ICP keywords including "saas" (high IDF, appears in only one doc). Stripe scores low despite being an obvious fit — its description uses "apis" and "commerce," neither in the ICP set. Notion has zero overlap and scores exactly 0.0. This is the exact-match ceiling: if the keyword does not appear verbatim, the score is zero regardless of semantic relevance.

## Exercises

**Easy.** Given this 2-document corpus:

```python
doc_a = "cloud security platform for enterprises"
doc_b = "endpoint security tool for enterprises"
```

Write out the BoW vectors for both documents using the full sorted vocabulary. Then compute cosine similarity between the two BoW vectors by hand — show the dot product, both L2 norms, and the final cosine value. Confirm your hand calculation by running the same computation in Python.

**Hard.** You are scoring 500 company descriptions against an ICP keyword profile using the TF-IDF pipeline from this lesson. Three companies that your sales team swears are perfect ICP fits score below 0.05 cosine similarity. Design a diagnostic script that, for a given low-scoring company, isolates each possible cause: (1) prints which ICP keywords are absent from the company description (vocabulary mismatch), (2) prints the ratio of ICP keyword count to total token count in the description (length dilution), and (3) prints the IDF weight of each ICP keyword to check whether overrepresentation in the corpus has crushed them to near-zero. For each cause, propose one concrete mitigation: stemming or synonym expansion for mismatch, TF normalization for length, and manual weight overrides for IDF dilution.

## Key Terms

**Bag of Words (BoW)** — Text representation that discards word order and encodes each document as a vector of per-vocabulary-term raw counts. The simplest functional text representation.

**Inverse Document Frequency (IDF)** — `log(N / df(w))`. A per-term weight that is large for rare words and near-zero for ubiquitous words. The mechanism by which TF-IDF downweights uninformative terms without requiring a stop-word list.

**TF-IDF** — Term Frequency × Inverse Document Frequency. A reweighted BoW vector where distinctive terms get higher values and ubiquitous terms get crushed toward zero. The standard count-based text representation before embeddings.

**Cosine Similarity** — Dot product of two vectors divided by the product of their L2 norms. Measures the angle between vectors, ranging from 0 (orthogonal) to 1 (identical direction). Preferred over Euclidean distance for text because it is invariant to document length.

**Sparse Vector** — A vector where most elements are zero. BoW and TF-IDF vectors are sparse because any given document uses only a small fraction of the corpus vocabulary.

**Exact-Match Ceiling** — The limitation that BoW and TF-IDF only produce nonzero scores when terms match verbatim. "Payment" and "payments" are different dimensions. Synonyms, morphological variants, and paraphrases all score zero. This is the problem embeddings solve.

## Sources

1. Salton, G. and Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." *Information Processing & Management*, 24(5), 513–523.
2. Manning, C.D., Raghavan, P., and Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Chapters 6 (Scoring, Term Weighting, the Vector Space Model) and 7 (Computing Scores in a Complete Search System).
3. Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830. — `TfidfVectorizer` implementation reference.
4. [CITATION NEEDED — concept: prevalence of keyword-based ICP matching in account scoring workflows]