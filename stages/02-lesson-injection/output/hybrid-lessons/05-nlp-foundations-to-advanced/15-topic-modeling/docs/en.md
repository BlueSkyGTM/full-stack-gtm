# Topic Modeling — LDA and BERTopic

## Learning Objectives

1. Implement LDA on a document corpus and extract topic-word distributions with observable output
2. Implement BERTopic on the same corpus and compare topic representations to LDA
3. Evaluate topic quality using coherence scores and interpretability heuristics
4. Configure BERTopic pipeline components — embedding model, UMAP, HDBSCAN — and explain how each affects topic output
5. Apply topic modeling to GTM text corpora — support tickets, lost-deal notes, reviews — and extract actionable themes

## The Problem

You have 5,000 support tickets, 10,000 Glassdoor reviews, or 2,000 lost-deal notes. You need the recurring themes without reading every word. Nobody has labeled these documents. You don't even know how many categories exist. Topic modeling extracts latent thematic structure from unlabeled text corpora — this lesson covers two algorithms that solve the same problem with different mechanisms.

Latent Dirichlet Allocation (LDA, 2003) assumes each document is a mixture of topics and each topic is a distribution over words. It uses Dirichlet priors to control how sparse those mixtures are. Inference reverses a generative process to discover the hidden topic assignments. Input is bag-of-words — word order is destroyed before processing.

BERTopic (2022) runs a four-stage pipeline: transformer embeddings capture semantic similarity, UMAP reduces dimensionality, HDBSCAN clusters documents by density, and c-TF-IDF extracts representative terms per cluster. Word order matters only in step one — everything downstream is clustering and frequency counting.

The structural difference matters at production time. LDA assigns each document a probability distribution over all topics (soft clustering). BERTopic assigns each document to exactly one cluster, then derives topics from cluster membership. This means LDA handles long mixed-topic documents better. BERTopic handles short texts better because transformer embeddings carry semantic information that bag-of-words loses.

## The Concept

**LDA generative story.** Before inference, LDA defines a generative process — how documents are assumed to be "created" from topics. Each topic β_k is a probability distribution over the vocabulary. Each document has its own topic mixture θ_d. To generate word w in document d, first sample a topic z from θ_d, then sample a word from β_z. Inference reverses this: given the observed words, compute the posterior over θ_d (per-document topic mixtures) and β_k (per-topic word distributions).

The Dirichlet prior α controls how many topics each document uses. A low α (e.g., 0.01) pushes the mixture toward one or two dominant topics per document — sparse assignment. A high α spreads probability mass across many topics. The prior η does the same for word distributions within each topic: low η means each topic uses few words distinctly. Collapsed Gibbs sampling and variational Bayes are the two inference engines; sklearn uses variational Bayes by default, which is faster but less precise on small corpora.

LDA's bag-of-words assumption discards word order entirely. The sentence "pricing killed the deal" and "the deal killed pricing" produce identical input. For long documents where word frequency patterns are strong, this is acceptable. For short texts — tweets, ticket subjects, review titles — word overlap is too sparse for LDA to find stable structure.

```mermaid
flowchart TD
    A["Raw Documents"] --> B["Embedding Model\n(sentence-transformers)"]
    B --> C["UMAP\nDimensionality Reduction"]
    C --> D["HDBSCAN\nDensity-Based Clustering"]
    D --> E["c-TF-IDF\nClass-based Term Frequency"]
    E --> F["Topic Representations"]

    B -- "semantic similarity\nword order matters here" --> C
    C -- "reduce 384+ dims\nt