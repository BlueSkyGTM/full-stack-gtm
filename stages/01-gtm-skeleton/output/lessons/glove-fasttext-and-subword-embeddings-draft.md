# GloVe, FastText, and Subword Embeddings

## Beat 1: Hook — Why Local Context Isn't Enough

Word2Vec trains on local context windows, which means it captures co-occurrence patterns one sentence at a time. Two problems emerge: global statistical relationships between distant words get flattened, and any word absent from training data gets no vector at all. GloVe addresses the first problem by factorizing a global co-occurrence matrix; FastText addresses the second by decomposing words into character n-grams so unseen words still receive a vector.

## Beat 2: Concept — Co-occurrence Factorization and Subword Decomposition

GloVe constructs a full word-word co-occurrence matrix X, where X_ij counts how often word j appears within a fixed window of word i across the entire corpus. It then factorizes this matrix directly, optimizing a weighted least-squares objective on log(X_ij). FastText keeps the skip-gram architecture but represents each word as the sum of its character n-grams (typically length 3–6), so the vector for "running" is the sum of vectors for `<ru`, `run`, `unn`, `nni`, `nin`, `ing`, `ng>`, plus the whole-word vector `<running>`. This means "running" shares subword components with "run" and "runner" even if "running" never appeared in training.

## Beat 3: Mechanism — The Two Algorithms

**GloVe**: Build co-occurrence matrix X from corpus. For each word pair (i, j), minimize: w_i · w_j + b_i + b_j - log(X_ij), weighted by f(X_ij) where f is a ramp function that caps the influence of extremely frequent pairs. The resulting w_i vectors are the embeddings. Convergence is typically 15–50 iterations over the full matrix.

**FastText**: For each word w in training data, generate all character n-grams of length 3–6 with boundary markers. The word's representation is v_w = (1/|G_w|) Σ g ∈ G_w z_g, where z_g is the n-gram vector. Training uses the same negative-sampling objective as Word2Vec skip-gram, but the prediction is the sum of subword vectors. At inference, any string gets a vector — even if the exact word was never seen — because its n-grams were likely observed as parts of other words.

## Beat 4: Code — Loading and Comparing Embeddings

Load pre-trained GloVe (via gensim's conversion from text format) and FastText vectors, compute cosine similarity for matched word pairs, and generate vectors for deliberately out-of-vocabulary strings to demonstrate FastText's subword fallback. Print similarity scores and OOV handling to stdout.

*Exercise hooks:*
- **Easy**: Load both embedding sets, print the 5 nearest neighbors to "revenue" in each.
- **Medium**: Construct 5 domain-specific words not in GloVe's vocabulary (e.g., "collaborationtool"), confirm GloVe returns OOV, then show FastText's nearest neighbors for the same terms.
- **Hard**: Build a function that computes analogy accuracy (a:b :: c:d) across both embedding sets on 20 analogy pairs, prints accuracy per set, and identifies cases where subword information helps or hurts.

## Beat 5: Use It — Embedding Choice for Entity Matching

In enrichment workflows (Zone 02 — Enrich), you often need to match company names, product terms, or industry jargon that won't appear in generic pre-trained vocabularies. FastText's subword decomposition handles typos, compound words, and novel entities without retraining. Use FastText when your input contains user-generated text, company names with creative spellings, or domain-specific compounds. Use GloVe when your vocabulary is clean and fixed (e.g., matching against a standardized industry taxonomy) and you want deterministic, smaller embeddings with no OOV edge-case behavior.

*Exercise hooks:*
- **Easy**: Given a list of 10 company names with intentional typos, compute FastText vectors and find the closest match from a clean reference list.
- **Medium**: Build a deduplication function that takes a list of raw lead company names, embeds each with FastText, clusters by cosine similarity threshold, and prints canonical groups.
- **Hard**: Implement a hybrid matcher: use GloVe for standardized industry codes and FastText for raw company names, combine scores with a weighted sum, and evaluate precision/recall against a labeled dataset.

## Beat 6: Ship It — Dimensionality, Caching, and Retrieval

Pre-trained FastText vectors at 300 dimensions for 2M words are ~2GB on disk. For real-time matching endpoints, reduce to 50–100 dimensions via PCA or use the pre-trained 50-d variants, and cache the embedding matrix in memory as a NumPy memmap. For batch enrichment jobs, load full 300-d vectors and pre-compute an approximate nearest-neighbor index (FAISS or Annoy) over your target vocabulary to avoid O(n) cosine comparisons. FastText's `.bin` model files include n-gram vectors for OOV inference; GloVe `.txt` files do not — if you need OOV handling at inference, FastText is the only option without retraining.

*Exercise hooks:*
- **Easy**: Load FastText 300-d vectors, reduce to 50-d with PCA, and verify that nearest-neighbor rankings are preserved for 10 test words.
- **Medium**: Build a FAISS index over a 10K-word vocabulary using FastText embeddings, query with 100 OOV strings, and print latency per query versus brute-force cosine search.
- **Hard**: Create a production-ready embedding service: load FastText `.bin` model, expose a function that takes a string, returns its vector (handling OOV), and benchmarks throughput at 1000 requests/second with concurrent requests.

---

**Learning Objectives:**
1. Load pre-trained GloVe and FastText embeddings and compute cosine similarity between word pairs.
2. Compare GloVe and FastText behavior on in-vocabulary and out-of-vocabulary terms.
3. Explain how FastText's character n-gram decomposition produces vectors for unseen words.
4. Configure FastText dimensionality and retrieval strategy for a given latency budget.
5. Evaluate embedding quality using analogy tasks and domain-specific matching accuracy.