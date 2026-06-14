## Ship It

In a production GTM stack, this math runs inside three tools:

**Vector databases** (Pinecone, Weaviate, Qdrant) store pre-normalized company embeddings and compute approximate nearest-neighbor similarity at scale. The similarity operation is still a dot product, but the database optimizes it for millions of vectors using approximate search algorithms (HNSW, IVF). When you query "find me companies similar to my ICP," the database computes cosine similarity against every stored vector and returns ranked results.

**Clay's AI enrichment columns** wrap embedding generation and similarity scoring behind a UI. When you add a "Find Similar Companies" or "Score Fit" column, Clay embeds the input, retrieves stored embeddings, and runs cosine similarity. The waterfall enrichment that pulls data from Apollo, Clearbit, and LinkedIn, then scores each enriched account, is executing the matrix operations you built above — just orchestrated across API calls. [CITATION NEEDED — concept: Clay's specific embedding model and similarity computation]

**Custom Python scripts** in your GTM engineering workspace handle the cases where Clay's built-in columns are not enough. If you need to score 50,000 accounts against a multi-dimensional ICP (combining firmographics, technographics, and semantic similarity), you batch the computation with NumPy and write results back to Clay via webhook.

Here is a production-ready function that takes a batch of company embeddings and an ICP vector, returns ranked results with scores, and is the kind of code you would deploy in a webhook handler:

```python
import numpy as np

def rank_companies_against_icp(company_embeddings, company_ids, icp_vector, top_k=10):
    company_embeddings = np.array(company_embeddings)
    icp_vector = np.array(icp_vector)
    
    company_norms = np.linalg.norm(company_embeddings, axis=1, keepdims=True)
    icp_norm = np.linalg.norm(icp_vector)
    
    company_embeddings = company_embeddings / company_norms
    icp_vector = icp_vector / icp_norm
    
    similarities = company_embeddings @ icp_vector
    
    ranked = np.argsort(similarities)[::-1][:top_k]
    
    return [(company_ids[i], float(similarities[i])) for i in ranked]

np.random.seed(42)
test_embeddings = np.random.randn(100, 384)
test_ids = [f"apollo-company-{i:04d}" for i in range(100)]
test_icp = np.random.randn(384)

ranked = rank_companies_against_icp(test_embeddings, test_ids, test_icp, top_k=5)

print("Production ICP Ranking Output\n")
print(f"{'Rank':<6} {'Company ID':<25} {'Score':<10}")
print("-" * 41)
for rank, (company_id, score) in enumerate(ranked, 1):
    print(f"{rank:<6} {company_id:<25} {score:.4f}")

print(f"\nTotal companies scored: {len(test_embeddings)}")
print(f"Embedding dimensions: {test_embeddings.shape[1]}")
print(f"All scores in valid range [-1, 1]: {all(-1 <= s <= 1 for _, s in ranked)}")
```

This function handles edge cases (zero-norm vectors would cause division by zero — in production, add a small epsilon or filter them). The output format matches what you would write back to a Clay table or push to a Slack webhook for SDR review.