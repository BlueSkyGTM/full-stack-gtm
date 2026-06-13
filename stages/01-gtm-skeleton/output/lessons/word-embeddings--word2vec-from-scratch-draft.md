# Word Embeddings — Word2Vec from Scratch

---

## Beat 1: Hook

The words in your GTM data — job titles, company descriptions, technographics — are categorical tokens with no inherent relationship. "VP Engineering" and "Head of Engineering" share zero signal in raw form. Word2Vec converts these tokens into dense vectors where semantic similarity becomes geometric distance. This is the foundation for every text-based enrichment and matching pipeline that follows.

---

## Beat 2: Concept

**Mechanism:** A neural network with one hidden layer, no activation, trained to predict context words from a target word (skip-gram) or vice versa (CBOW). The hidden layer weights become the embeddings. The network never matters at inference — only the weight matrix does.

Cover:
- Co-occurrence statistics as the signal source
- Skip-gram architecture: input word → hidden layer → output softmax over vocabulary
- Why softmax over full vocab is O(|V|) and negative sampling reduces this to O(k)
- The weight matrix as the lookup table — each row is one word's embedding
- Why cosine similarity, not Euclidean distance, measures semantic closeness in high dimensions

---

## Beat 3: Demonstration

Build skip-gram with negative sampling on a small corpus, print the resulting vectors and nearest neighbors.

Observable output:
- Training loss decreasing over epochs
- Cosine similarity matrix for a small vocabulary showing semantically similar words clustering
- A few nearest-neighbor queries: "king" neighbors, "engineer" neighbors

---

## Beat 4: Use It

**GTM Redirect:** Semantic matching for company/title enrichment — Zone 2 enrichment pipelines. When you embed job titles and company descriptions, you can cluster, deduplicate, and match without exact string overlap. This feeds directly into ICP scoring and account deduplication workflows.

Exercise hooks:
- **Easy:** Train embeddings on a corpus of job titles, print the 5 nearest neighbors for "senior engineer"
- **Medium:** Given two lists of company descriptions (different CRM fields), compute pairwise cosine similarity and flag duplicates above a threshold
- **Hard:** Build a title normalization function that maps raw titles to a canonical set using embedding distance, test against a held-out mapping

---

## Beat 5: Ship It

**GTM Redirect:** Foundational for Zone 2 enrichment and Zone 3 personalization. Embeddings power title normalization, company deduplication, and intent signal clustering. Without this, downstream classification and matching rely on exact matches or brittle regex.

Exercise hooks:
- **Easy:** Export trained embeddings to a JSON file, write a lookup function that returns neighbors
- **Medium:** Wrap the training pipeline in a function that takes raw text, a vocab size cap, and embedding dimension, returns a reusable embedding dict
- **Hard:** Implement a caching layer — retrain only when new vocabulary exceeds a threshold — and benchmark retrieval latency for 10K similarity queries

---

## Beat 6: Evaluate

Exercise hooks:
- **Easy:** Given a trained embedding, compute cosine similarity between two word pairs and print which pair is more similar
- **Medium:** Implement an analogy test (a:b :: c:?) using vector arithmetic, evaluate accuracy against a small test set
- **Hard:** Compare skip-gram vs. CBOW on the same corpus using intrinsic evaluation (similarity benchmark), report which produces higher-quality embeddings for short-text GTM data

---

## Learning Objectives

1. Implement skip-gram with negative sampling from raw text using only NumPy
2. Train a word-to-vector mapping via gradient descent on a co-occurrence objective
3. Compute cosine similarity between learned embeddings and retrieve nearest neighbors
4. Evaluate embedding quality using analogy tests and similarity benchmarks
5. Compare skip-gram and CBOW architectures on short-form GTM text data