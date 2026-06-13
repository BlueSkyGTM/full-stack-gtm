# Image Retrieval & Metric Learning

## Beat 1: Hook

You have 50,000 product images and a new photo from a prospect. Find the match. Flat pixel comparison dies at scale — you need embeddings where distance = similarity. Metric learning is the mechanism that makes visual search work.

## Beat 2: Core Concept

**Mechanism**: Train a network to map inputs into an embedding space where Euclidean or cosine distance reflects semantic similarity, not pixel overlap. The network learns from *pairs* or *triplets* of examples — same-category items pulled together, different-category items pushed apart. Loss functions (contrastive, triplet, ArcFace) define the pushing/pulling geometry. At inference, you embed a query image, run approximate nearest neighbor (ANN) search against a pre-built index, and return top-k matches.

**Key algorithms to cover**:
- Siamese networks with contrastive loss
- Triplet loss (anchor, positive, negative)
- ArcFace / CosFace angular margin penalties
- Retrieval pipeline: embed → index → ANN search

**Tools** (named after mechanism): PyTorch Metric Learning library implements these loss functions; FAISS builds the ANN index; CLIP provides a pre-trained embedding baseline.

## Beat 3: Implement It

Build a working image retrieval pipeline end-to-end:

1. **Easy exercise**: Compute embeddings from a pre-trained ResNet, build a FAISS L2 index, run a query, print top-5 distances and indices. Observable output: ranked distance list.
2. **Medium exercise**: Implement triplet loss from scratch on a toy dataset (random tensors as image proxies). Print loss values per epoch to confirm convergence. Observable output: decreasing loss curve numbers.
3. **Hard exercise**: Fine-tune a small CNN on a labeled dataset (e.g., Omniglot or a synthetic set) using triplet sampling, embed all test images, compute recall@k. Observable output: recall metric printed to terminal.

All code runs in terminal. No browser. Print statements confirm every result.

## Beat 4: Use It

**GTM Redirect**: Logo and brand detection in prospect assets — Zone 01 (ICP & Targeting). Given a screenshot of a prospect's website or social feed, retrieve matching company logos from a pre-indexed brand library. This is similarity search over logo embeddings, not keyword matching.

[CITATION NEEDED — concept: logo detection via embedding retrieval for account identification]

**Mechanism mapping**: The same ANN retrieval pipeline from Beat 3 applies. Replace "product images" with "logo crops." Index logo embeddings from your ICP list. Embed the prospect's asset. Run the search. The metric learning model determines whether "similarity" captures brand variation (color shifts, resizing) or collapses it.

**Exercise hook (medium)**: Index 10 synthetic logo-like images (colored shapes with text), query with a modified version, print the top-3 matches with cosine distances. Observable: correct logo retrieved despite color shift.

## Beat 5: Ship It

**Deployment considerations**:
- Embedding computation is a batch job; ANN index serves queries at latency proportional to index size and algorithm (IVF, HNSW)
- Index versioning: when you add new images, re-index or append. FAISS supports additive indices.
- Distance thresholds: what cosine similarity counts as a "match"? This is a business decision, not an ML decision. Ship with a configurable threshold and log false positives.

**Exercise hook (hard)**: Build a CLI tool that takes a directory of images, builds a FAISS index, saves it to disk, then loads it in a separate process and answers a query. Print index size, query latency in ms, and top-k results. Observable: end-to-end retrieval with timing.

## Beat 6: Extend It

**Where to go deeper**:
- Multi-modal retrieval: CLIP joint text-image embeddings for querying images with natural language descriptions
- Hard negative mining: smarter triplet sampling strategies that improve embedding quality
- Vector databases: Milvus, Qdrant, Pinecone as managed ANN alternatives to FAISS — same mechanism, different operational model
- Evaluation: mean Average Precision (mAP), recall@k, precision-recall curves for retrieval systems

**Redirect**: Multi-modal retrieval connects to Zone 02 (Signal Detection) — retrieving relevant prior interactions or content using mixed text/image queries. [CITATION NEEDED — concept: multi-modal retrieval for GTM signal enrichment]

---

## Learning Objectives (testable)

1. **Implement** triplet loss and contrastive loss on sample tensor data, printing convergence evidence.
2. **Build** a FAISS index from image embeddings and execute a nearest-neighbor query with observable ranked output.
3. **Compare** cosine distance vs. L2 distance for retrieval quality on a controlled dataset.
4. **Configure** a retrieval threshold for binary match/no-match decisions and articulate the precision-recall tradeoff.
5. **Evaluate** a retrieval pipeline using recall@k on a labeled test set.