## Ship It

Chunking failures account for most QA errors in production — not model failures, not embedding failures. A chunk that splits a sentence in half, a chunk that combines two unrelated topics, a chunk that is too small to carry enough context for the LLM to generate a correct answer. Before deploying a QA system, measure where failures occur. The three guardrails below address the most common production issues: latency visibility, confidence thresholds, and evaluation logging.

Latency matters because retrieval and generation happen on every query. A vector similarity search over 10,000 chunks is fast (milliseconds). An LLM call adds 500-2000ms. If you do not measure these separately, you cannot optimize them — and you cannot tell a rep whether the system will respond in 1 second or 10 seconds.

```python
import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

documents = [
    "Apple's primary revenue model is hardware sales, with the iPhone generating approximately 52% of total revenue in fiscal year 2023.",
    "Apple's services segment, including App Store, iCloud, and Apple Music, generated $85 billion in revenue in fiscal year 2023.",
    "Tesla's automotive revenue reached $82 billion in 2023, with the Model Y being the best-selling vehicle globally.",
    "Tesla's energy storage business deployed 14.7 GWh of capacity in 2023, a 125% increase year-over-year.",
    "Microsoft's cloud business, Azure, grew 29% year-over-year in Q3 2024, contributing significantly to overall revenue.",
    "Microsoft's productivity segment, including Office 365 and LinkedIn, generated $77.7 billion in revenue in fiscal year 2023.",
]

CONFIDENCE_THRESHOLD = 0.35

model = SentenceTransformer('all-MiniLM-L6-v2')

t0 = time.perf_counter()
embeddings = model.encode(documents, normalize_embeddings=True)
embed_time = time.perf_counter() - t0

query = "What is Tesla's energy storage business?"
t1 = time.perf_counter()
query_embedding = model.encode([query], normalize_embeddings=True)[0]
similarities = np.dot(embeddings, query_embedding)
retrieve_time = time.perf_counter() - t1

top_k = 3
top_indices = np.argsort(similarities)[::-1][:top_k]
retrieved = [(documents[i], float(similarities[i])) for i in top_indices]

best_score = retrieved[0][1]
print(f"Latency breakdown:")
print(f"  Embed all docs: {embed_time*1000:.1f}ms")
print(f"  Embed query + retrieve: {retrieve_time*1000:.1f}ms")
print(f"  Best similarity score: {best_score:.4f}")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")

if best_score < CONFIDENCE_THRESHOLD:
    print("\n  RESULT: No relevant information found (below threshold).")
    answer = "No relevant information found."
    sources = []
else:
    context = "\n".join([f"[{i+1}] {chunk}" for i, (chunk, _) in enumerate(retrieved)])
    prompt = f"""Based on the following context, answer the question. Cite the source number. If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    
    t2 = time.perf_counter()
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content
    else:
        answer = "[LLM would generate answer from retrieved context]"
    gen_time = time.perf_counter() - t2
    
    print(f"  LLM generation: {gen_time*1000:.1f}ms")
    print(f"\n  ANSWER: {answer}")
    sources = [i+1 for i in top_indices[:3]]

log_entry = {
    "query": query,
    "retrieved_chunks": [{"chunk": c, "score": s} for c, s in retrieved],
    "generated_answer": answer,
    "sources": sources,
    "best_score": best_score,
    "below_threshold": best_score < CONFIDENCE_THRESHOLD,
}

with open("qa_eval_log.jsonl", "a") as f:
    f.write(json.dumps(log_entry) + "\n")

print(f"\n  Logged to qa_eval_log.jsonl")
```

The confidence threshold is the most important guardrail. Without it, the system will generate an answer from whatever chunk has the highest similarity score — even if that score is 0.15, meaning the chunk is barely related. Setting a threshold of 0.35 (adjust based on your embedding model and corpus) forces the system to admit ignorance rather than hallucinate from weak context. The JSONL log creates an audit trail: every query, its retrieved chunks, the generated answer, and the similarity scores. Review this log periodically to find queries that retrieved the wrong chunks — those are the cases where chunking or embedding needs adjustment.

Index refresh strategy is the other production concern. Your research corpus is not static — new 10-Ks are filed, new transcripts are published, new notes are added. The embedding index must be rebuilt or incrementally updated when the source documents change. A simple approach: store documents with a timestamp, re-embed only new or modified documents, and append to the vector store. A more robust approach: version the entire index and rebuild from scratch on a schedule (nightly, weekly) to avoid drift from incremental updates that accumulate errors.