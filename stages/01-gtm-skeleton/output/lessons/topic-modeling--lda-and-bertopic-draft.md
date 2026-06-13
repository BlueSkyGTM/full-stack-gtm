# Topic Modeling — LDA and BERTopic

## Hook It
You have 5,000 support tickets, 10,000 Glassdoor reviews, or 2,000 lost-deal notes. You need the recurring themes without reading every word. Topic modeling extracts latent thematic structure from text corpora — this lesson covers two algorithms that solve the same problem with different mechanisms.

## Ground It
**LDA mechanism**: A generative probabilistic model where each document is a mixture of topics, and each topic is a distribution over words. Uses Dirichlet priors to control sparsity. Inference reverses the generative process to discover topic assignments. Bag-of-words input — word order is destroyed.

**BERTopic mechanism**: A pipeline of four steps — transformer embeddings to capture semantic similarity, UMAP for dimensionality reduction, HDBSCAN for density-based clustering, and c-TF-IDF to extract representative terms per cluster. Word order matters in step one; everything after is clustering.

**Key difference**: LDA assumes documents are mixtures of topics (soft clustering baked in). BERTopic assigns each document to one cluster, then derives topics from cluster membership. LDA struggles with short texts. BERTopic struggles without enough documents for density estimation.

**Learning Objectives**:
1. Implement LDA on a corpus and extract topic-word distributions with observable output
2. Implement BERTopic on the same corpus and compare topic representations to LDA
3. Evaluate topic quality using coherence scores and interpretability heuristics
4. Configure BERTopic pipeline components (embedding model, UMAP parameters, HDBSCAN parameters)
5. Apply topic modeling to GTM text data and extract actionable themes

## Show It
**Code Block 1 — LDA from scratch on a synthetic corpus**: Build a document-term matrix with `sklearn` CountVectorizer, fit `LatentDirichletAllocation`, print top words per topic, print per-document topic distributions. Output confirms topics were extracted and documents have mixture weights.

**Code Block 2 — BERTopic on the same corpus**: Generate sentence embeddings with `sentence-transformers`, run `BERTopic` pipeline, print topics with representative words, print per-document topic assignments. Output confirms clustering and topic extraction worked.

**Code Block 3 — Coherence comparison**: Compute topic coherence for both models using `gensim` CoherenceModel. Print scores side by side. Output confirms one model scores higher on this specific corpus — demonstrates evaluation, not superiority.

**Mechanism breakdown for each code block**: Explain what the algorithm did before showing the tool call. "The Dirichlet prior controls how many topics each document uses — a low alpha forces sparse topic mixtures" precedes `LatentDirichletAllocation(doc_topic_prior=0.01)`.

## Build It
Practitioner builds both models on a real dataset and compares results.

**Exercise hooks**:

- **Easy**: Run LDA on the provided dataset of 500 customer reviews. Print top 8 words for each of 5 topics. Identify which topic maps to pricing complaints vs. feature requests by reading the top words.
- **Medium**: Run BERTopic on the same 500 reviews. Adjust `min_cluster_size` in HDBSCAN to merge small clusters. Compare topic count to LDA. Print coherence scores for both and state which model produced more interpretable topics for this dataset.
- **Hard**: Build a pipeline that runs both LDA and BERTopic on the dataset, computes coherence, and prints a comparison table. Add a function that takes a new document and assigns it to the closest existing topic in both models. Print the assigned topic and top words for verification.

## Use It
**GTM Redirect**: Topic modeling maps to the **Voice of Customer** cluster and **ICP Research** cluster. Apply to: lost-deal notes to surface recurring objection themes, job postings to extract skill/theme clusters for persona targeting, support tickets to identify feature-gap patterns by segment.

**Exercise hooks**:

- **Easy**: Run BERTopic on 200 lost-deal notes (provided dataset). Print the 3 largest topics. For each topic, write one sentence describing the objection theme — this is your VoC signal.
- **Medium**: Topic-model 500 job postings from a target vertical. Extract the top 5 topic clusters. Map each cluster to a pain point or responsibility. Output a table: topic label → top words → GTM pain point hypothesis.
- **Hard**: Build a topic comparison pipeline across two segments (e.g., enterprise vs. mid-market job postings). Run BERTopic separately on each segment. Print topics unique to each segment. Output actionable insight: "Enterprise segment mentions [topic] 3x more than mid-market."

## Ship It
**Production considerations**: Topic models degrade on domain shift — retrain quarterly or when new vocabulary appears. LDA is fast and interpretable but weak on short texts and semantic nuance. BERTopic captures semantics but requires GPU for embedding at scale and needs minimum document counts for HDBSCAN density estimation.

**Evaluation checklist**: Coherence score is a proxy, not ground truth. Manual inspection of top words per topic is still required. If you can't label a topic in under 5 seconds, it's probably not a clean topic.

**When to pick which**:
- Short texts (<50 words), large corpus: LDA struggles, BERTopic works
- Need per-document topic mixtures: LDA
- Need reproducible, fast training on CPU: LDA
- Need semantic clustering with minimal preprocessing: BERTopic

**GTM Redirect**: Use BERTopic for VoC analysis on support tickets and lost-deal notes — the semantic embeddings handle the messy language in customer feedback better than bag-of-words. Use LDA for large-scale content auditing (blog posts, case studies) where you need to check topic coverage across your content library.

**Exercise hooks**:

- **Easy**: Take the BERTopic model from "Use It" and write a function that flags new incoming support tickets with a topic label and confidence score. Print output for 5 test tickets.
- **Medium**: Build a topic monitoring script that re-runs BERTopic weekly on a growing dataset, tracks topic prevalence over time, and prints "emerging topic" alerts when a new cluster appears that wasn't in the previous run.
- **Hard**: Deploy a topic model as a CLI tool that accepts a CSV of text data, auto-selects LDA or BERTopic based on text length distribution (mean < 50 words → BERTopic, else → LDA), and outputs a topic summary report with coherence scores and top documents per topic.