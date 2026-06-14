## Ship It

Production embedding systems fail on four dimensions that the demo code above ignores entirely. First, model versioning: when Voyage releases Embed v4 or BGE ships M4, switching models means every vector in your index is invalid. The old vectors and new vectors live in different geometric spaces — cosine similarity across them is meaningless. You must re-embed the entire corpus and swap atomically, or run dual indexes during migration. Pin your model version in configuration and treat re-embedding as a migration with a rollback plan.

Second, batching. The `model.encode()` calls above embed one or twelve sentences at a time. In production, you batch 64-256 passages per API call to amortize network overhead. For hosted APIs, batching also determines your cost — most providers charge per token regardless of batch size, but round-trip latency drops linearly with batch size up to the provider's max batch limit. For local models, GPU memory is the constraint: a 7B-parameter embedding model processing 256 passages at 512 tokens each needs ~8GB of VRAM for the attention matrices alone.

Third, storage and retrieval. Embeddings are not a database. They are a similarity index that tells you which vectors are close to your query vector. They do not filter by date, region, company size, or any other metadata your GTM stack depends on. Every production vector store pairs vector search with metadata filtering: pre-filter the candidate set by metadata (e.g., "leads from US companies, created in last 30 days"), then run vector search on the filtered set. This is why sqlite-vss, pgvector, Pinecone, and Weaviate all support metadata pre-filtering — without it, you retrieve semantically similar leads from 2023 that are completely irrelevant.

```python
import sqlite3
import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = sqlite3.connect(":memory:")
conn.execute("""
    CREATE TABLE leads (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        region TEXT NOT NULL,
        created_days_ago INTEGER NOT NULL,
        embedding BLOB NOT NULL
    )
""")

leads = [
    ("I want to see a demo of the analytics dashboard", "US", 2),
    ("How does your API handle authentication?", "US", 5),
    ("Interested in a reseller partnership in Germany", "EU", 1),
    ("Can we schedule a product walkthrough?", "US", 3),
    ("What are the rate limits on enterprise tier?", "EU", 7),
    ("We want to co-sell with your team in EMEA", "EU", 4),
    ("Show me the reporting features", "US", 1),
    ("Does the API support bulk export?", "APAC", 10),
    ("Looking for a technology partnership", "APAC", 2),
    ("We need a demo for our leadership team", "US", 6),
    ("How does webhook delivery work?", "EU", 3),
    ("Can our agency become a certified partner?", "US", 8),
    ("Pricing for 500 seats", "US", 1),
    ("What does the enterprise plan cost?", "EU", 2),
    ("Investment required for a 200-seat rollout", "APAC", 5),
]

texts = [l[0] for l in leads]
embeddings = model.encode(texts, normalize_embeddings=True)

for i, (text, region, days_ago) in enumerate(leads):
    emb_bytes = embeddings[i].astype(np.float32).tobytes()
    conn.execute(
        "INSERT INTO leads (id, text, region, created_days_ago, embedding) VALUES (?, ?, ?, ?, ?)",
        (i, text, region, days_ago, emb_bytes)
    )
conn.commit()
print(f"Inserted {len(leads)} leads into SQLite")
print()

def vector_search(query, limit=5, region_filter=None, max_age_days=None):
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    
    sql = "SELECT id, text, region, created_days_ago, embedding FROM leads"
    conditions = []
    params = []
    
    if region_filter:
        conditions.append("region = ?")
        params.append(region_filter)
    if max_age_days:
        conditions.append("created_days_ago <= ?")
        params.append(max_age_days)
    
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    
    rows = conn.execute(sql, params).fetchall()
    
    results = []
    for row in rows:
        lead_emb = np.frombuffer(row[4], dtype=np.float32)
        score = float(np.dot(query_emb, lead_emb))
        results.append((score, row[0], row[1], row[2], row[3]))
    
    results.sort(key=lambda x: -x[0])
    return results[:limit], len(rows)

query = "I want to see the product"

print("=" * 75)
print(f"QUERY: '{query}'")
print("=" * 75)

print("\n[1] NO METADATA FILTER — all 15 leads scanned")
t0 = time.perf_counter()
results, scanned = vector_search(query, limit=5)
elapsed = (time.perf_counter() - t0) * 1000
print(f"    Scanned {scanned} records in {elapsed:.2f}ms")
for score, lead_id, text, region, age in results:
    print(f"    [{score:.4f}] #{lead_id} ({region}, {age}d ago) {text}")

print("\n[2] METADATA PRE-FILTER: region=EU, age<=5 days")
t0 = time.perf_counter()
results, scanned = vector_search(query, limit=5, region_filter="EU", max_age_days=5)
elapsed = (time.perf_counter() - t0) * 1000
print(f"    Scanned {scanned} records in {elapsed:.2f}ms")
for score, lead_id, text, region, age in results:
    print(f"    [{score:.4f}] #{lead_id} ({region}, {age}d ago) {text}")

print("\n[3] METADATA PRE-FILTER: region=US, age<=3 days")
t0 = time.perf_counter()
results, scanned = vector_search(query, limit=5, region_filter="US", max_age_days=3)
elapsed = (time.perf_counter() - t0) * 1000
print(f"    Scanned {scanned} records in {elapsed:.2f}ms")
for score, lead_id, text, region, age in results:
    print(f"    [{score:.4f}] #{lead_id} ({region}, {age}d ago) {text}")

print("\n" + "=" * 75)
print("STORAGE ANALYSIS")
print("=" * 75)
vec_dim = embeddings.shape[1]
bytes_per_vec = vec_dim * 4
n_leads = len(leads)
print(f"Vector dimension: {vec_dim} (float32)")
print(f"Bytes per vector: {bytes_per_vec}")
print(f"Total vectors: {n_leads}")
print(f"Total embedding storage: {n_leads * bytes_per_vec:,} bytes ({n_leads * bytes_per_vec / 1024:.1f} KB)")
print(f"At 100k leads: {100000 * bytes_per_vec / 1024 / 1024:.1f} MB")
print(f"At 1M leads: {1000000 * bytes_per_vec / 1024 / 1024 / 1024:.2f} GB")
print()
print("With matryoshka truncation to 256 dims (if model supported):")
mat_bytes = 256 * 4
print(f"  Bytes per vector: {mat_bytes}")
print(f"  Storage savings: {(1 - mat_bytes/bytes_per_vec)*100:.0f}%")
print(f"  At 1M leads: {1000000 * mat_bytes / 1024 / 1024