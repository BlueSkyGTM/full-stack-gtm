## Ship It

The script below is a standalone retrieval service. It ingests a directory of text files, builds a persistent hybrid index (BM25 token state plus a `.npy` embedding matrix), exposes a CLI query interface, and logs every query and result set to JSONL for offline evaluation.

```python
import argparse
import json
import os
import pickle
import numpy as np
from datetime import datetime
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_DIR = ".retrieval_index"
MODEL_NAME = "all-MiniLM-L6-v2"

def chunk_text(text, max_tokens=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks

def build_index(input_dir):
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    for filename in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": filename,
                "chunk_idx": idx,
                "text": chunk,
            })

    texts = [c["text"] for c in all_chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    embeddings = model.encode(texts, normalize_embeddings=True)

    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    print(f"Indexed {len(all_chunks)} chunks from {len(set(c['source'] for c in all_chunks))} files.")
    for source in sorted(set(c["source"] for c in all_chunks)):
        count = sum(1 for c in all_chunks if c["source"] == source)
        print(f"  {source}: {count} chunks")
    return all_chunks

def load_index():
    with open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "r") as f:
        chunks = json.load(f)
    return bm25, embeddings, chunks

def bm25_search(bm25, chunks, query, k=10):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    order = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in order]

def dense_search(embeddings, chunks, model, query, k=10):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    order = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in order]

def rrf_merge(lex, dense, k=60, top_n=10):
    rrf = {}
    meta = {}
    for rank, (chunk, _) in enumerate(lex):
        key = f"{chunk['source']}:{chunk['chunk_idx']}"
        rrf[key] = rrf.get(key, 0) + 1.0 / (k + rank + 1)
        meta[key] = chunk
    for rank, (chunk, _) in enumerate(dense):
        key = f"{chunk['source']}:{chunk['chunk_idx']}"
        rrf[key] = rrf.get(key, 0) + 1.0 / (k + rank + 1)
        meta[key] = chunk
    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(meta[key], score) for key, score in ranked]

def query_index(query, top_k=5, log_file="queries.jsonl"):
    if not os.path.exists(os.path.join(INDEX_DIR, "bm25.pkl")):
        print("No index found. Run: python retrieve.py --build <directory>")
        return

    bm25, embeddings, chunks = load_index()
    model = SentenceTransformer(MODEL_NAME)

    lex = bm25_search(bm25, chunks, query, k=top_k * 2)
    dense = dense_search(embeddings, chunks, model, query, k=top_k * 2)
    hybrid = rrf_merge(lex, dense, top_n=top_k)

    print(f"\nQUERY: {query}")
    print(f"{'='*80}")
    for chunk, score in hybrid:
        bm25_s = next((s for c, s in lex if c["source"] == chunk["source"] and c["chunk_idx"] == chunk["chunk_idx"]), 0.0)
        dense_s = next((s for c, s in dense if c["source"] == chunk["source"] and c["chunk_idx"] == chunk["chunk_idx"]), 0.0)
        print(f"\n  [{chunk['source']} chunk={chunk['chunk_idx']}] RRF={score:.4f} BM25={bm25_s:.4f} Dense={dense_s:.4f}")
        print(f"  {chunk['text'][:200]}")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "top_k": top_k,
        "results": [
            {
                "source": c["source"],
                "chunk_idx": c["chunk_idx"],
                "rrf_score": s,
                "bm25_score": bm25_s,
                "dense_score": dense_s,
                "text": c["text"][:300],
            }
            for c, s in hybrid
        ],
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def make_sample_docs(output_dir="sample_docs"):
    os.makedirs(output_dir, exist_ok=True)
    samples = {
        "acme_10k.txt": "Acme Corporation reported annual recurring revenue of $84M, representing 52% year-over-year growth. The company expanded into the EMEA region with offices in London and Munich. Net revenue retention reached 124% driven by upsell of the premium platform tier. The SX-4400-T hardware module was deprecated in favor of the SX-4500-T next-generation processor. Customer retention initiatives including proactive success management reduced gross churn from 11% to 6%.",
        "acme_press.txt": "Acme Corporation today announced its European market expansion strategy, targeting $20M in new ARR within 18 months. The EMEA expansion includes dedicated sales teams in London, Frankfurt, and Paris. CEO Jane Doe cited strong demand from DACH customers as a key driver. The company also launched localized product offerings for the European market.",
        "acme_earnings.txt": "On the earnings call, CFO John Smith highlighted that net dollar churn improved to negative 3%, meaning expanding customers more than offset churned revenue. The monetization shift from perpetual licensing to subscription pricing continued to drive predictable recurring revenue. Competitive displacement against VendorX resulted in 34 won deals in the enterprise segment.",
    }
    for filename, content in samples.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(content)
    print(f"Sample documents written to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid retrieval service")
    parser.add_argument("--build", metavar="DIR", help="Build index from text files in directory")
    parser.add_argument("--sample", action="store_true", help="Generate sample documents")
    parser.add_argument("--query", metavar="TEXT", help="Query the index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    if args.sample:
        make_sample_docs()
    elif args.build:
        build_index(args.build)
    elif args.query:
        query_index(args.query, top_k=args.top_k)
    else:
        parser.print_help()
```

Run the full pipeline in three commands:

```bash
python retrieve.py --sample
python retrieve.py --build sample_docs
python retrieve.py --query "EMEA expansion targets" --top-k 3
```

The output prints each result with its source file, chunk index, and three scores: BM25, dense cosine similarity, and the fused RRF score. The JSONL log captures every query with timestamps and full result metadata — this is the evaluation substrate you need to measure recall@k over time as you tune chunk size, overlap, and the RRF `k` parameter.