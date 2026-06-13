# Bag of Words, TF-IDF, and Text Representation

## Learning Objectives

1. Build a Bag of Words vector from raw text using term counting
2. Calculate TF-IDF weights by hand and confirm with scikit-learn
3. Compare two documents using cosine similarity on TF-IDF vectors
4. Rank a list of company descriptions against an ICP keyword profile using text similarity

---

## Beat 1: Hook

Why machines can't read — and what we do instead. Text must become vectors before any classifier, scorer, or clustering algorithm can touch it. Introduces the core problem of text representation: discrete symbols → fixed-length numeric arrays.

---

## Beat 2: Concept

**Bag of Words mechanism**: tokenize, build vocabulary across corpus, count term occurrences per document → sparse vector. Demonstrates why raw counts break down: "the" dominates every document.

**TF-IDF mechanism**: multiply term frequency by inverse document frequency (log of total docs divided by docs containing term). Downweights ubiquitous terms, upweights distinctive ones. Shows the formula and walks through a 3-document example by hand.

**Cosine similarity**: dot product of normalized vectors. Why cosine over Euclidean for text (document length invariance).

---

## Beat 3: Code

Three working scripts, each with observable output:

1. **BoW from scratch** — build vocabulary, count terms, print sparse vectors for a 3-document corpus
2. **TF-IDF from scratch** — compute IDF weights, multiply by TF, print weighted vectors; then confirm outputs match `sklearn.feature_extraction.text.TfidfVectorizer` on same corpus
3. **Cosine similarity** — rank documents against a query vector, print ranked list with scores

---

## Beat 4: Use It

**GTM Redirect → Zone 1: ICP Fit Scoring**

Given a list of company descriptions (simulated enrichment data) and an ICP profile described as a keyword-weighted query, compute TF-IDF vectors for all companies, calculate cosine similarity against the ICP query, and rank accounts by fit score. This is the text-representation primitive behind any keyword-based ICP matching in account scoring workflows.

---

## Beat 5: Ship It

Wrap the ranking logic into a CLI script that reads company descriptions from a JSON file and outputs a ranked CSV. Covers sparse matrix handling with `scipy.sparse`, vocabulary persistence with `pickle`, and the scaling limit: vocabulary size grows with corpus size, TF-IDF recomputes when corpus changes — point to embeddings (future lesson) as the evolution past this ceiling.

---

## Beat 6: Evaluate

- **Easy**: Given a 2-document corpus and a vocabulary, write out the BoW vectors and compute cosine similarity
- **Medium**: Add a new document to an existing corpus — which IDF values change and which stay the same? Explain why
- **Hard**: A company description scores 0.03 cosine similarity against an ICP query. Diagnose whether the problem is vocabulary mismatch, document length, or IDF weighting — and propose a concrete fix

---

## GTM Redirect Rules

- **Beat 4 & Beat 5 redirect**: Zone 1 (ICP Fit Scoring) — TF-IDF + cosine similarity is the mechanism behind keyword-based account ranking. This is not a metaphor; this is the actual math used in text-matching enrichment workflows.
- **No forced connection**: If the practitioner is not in a text-heavy scoring context, the redirect is "foundational for any pipeline that classifies, clusters, or ranks unstructured text" — no fabricated GTM scenario.