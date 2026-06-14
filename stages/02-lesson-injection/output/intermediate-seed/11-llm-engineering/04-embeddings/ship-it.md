## Ship It

Production embedding pipelines fail in three predictable ways: you re-embed unchanged records and pay for it, you batch one-at-a-time and add 50× latency, or the provider silently updates model weights and your similarity thresholds drift. None of these are exotic. All three will hit you in the first week of production.

Batch embedding is the lowest-effort, highest-impact fix. Embedding models are optimized for batch input — the GPU/CPU amortizes the forward pass across multiple texts. Calling `.encode()` on individual strings in a loop is the most common production mistake. Measure it:

```python
from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [f"inbound message number {i} about topic {i % 5}" for i in range(50)]

start = time.perf_counter()
for text in texts:
    _ = model.encode([text])
loop_time = time.perf_counter() - start

start = time.perf_counter()
_ = model.encode(texts)
batch_time = time.perf_counter() - start

print(f"50 texts, one-at-a-time: {loop_time:.3f}s ({loop_time / 50 * 1000:.1f} ms/item)")
print(f"50 texts, single batch:  {batch_time:.3f}s ({batch_time / 50 * 1000:.1f} ms/item)")
print(f"Speedup: {loop_time / batch_time:.1f}x")
```

Caching prevents re-embedding. If a record's text hasn't changed, its vector hasn't changed. Hash the input text, store the vector keyed by that hash, and skip the model call on cache hits. This matters because embedding is the most expensive part of a semantic search pipeline — the dot-product search itself is nearly free by comparison:

```python
from sentence_transformers import SentenceTransformer
import hashlib
import json
import os
import time
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_FILE = "embedding_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def embed_with_cache(texts, model_name="all-MiniLM-L6-v2"):
    cache = load_cache()
    cache_key_prefix = f"{model_name}:"
    results = []
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = cache_key_prefix + text_hash
        if key in cache:
            results.append((i, cache[key]))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    cache_hits = len(results)
    cache_misses = len(uncached_texts)

    if uncached_texts:
        new_embeddings = model.encode(uncached_texts)
        for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            key = cache_key_prefix + text_hash
            cache[key] = emb.tolist()
            results.append((idx, emb.tolist()))
        save_cache(cache)

    results.sort(key=lambda x: x[0])
    return np.array([r[1] for r in results]), cache_hits, cache_misses

texts = [f"inbound lead message about {topic}" for topic in [
    "pipeline growth", "technical integration", "pricing question",
    "demo request", "security review",
]]

if os.path.exists(CACHE_FILE):
    os.remove(CACHE_FILE)

print("=== First run (cold cache) ===")
start = time.perf_counter()
embs, hits, misses = embed_with_cache(texts)
elapsed = time.perf_counter() - start
print(f"Cache hits: {hits}, misses: {misses}")
print(f"Time: {elapsed:.3f}s")

print("\n=== Second run (warm cache) ===")
start = time.perf_counter()
embs, hits, misses = embed_with_cache(texts)
elapsed = time.perf_counter() - start
print(f"Cache hits: {hits}, misses: {misses}")
print(f"Time: {elapsed:.3f}s")
print(f"Speedup from cache: {'significant' if misses == 0 else 'none — still had misses'}")

print(f"\nEmbedding shape: {embs.shape}")

os.remove(CACHE_FILE)
```

Model version pinning is the silent killer. When an embedding provider updates weights — even a "minor" update — every vector in your database becomes stale relative to the new model. Queries embedded with the new weights will have different cosine similarity relationships against vectors embedded with the old weights. Your thresholds (e.g., "route to sales if cosine > 0.7") will silently break. This is why your embedding store must record which model produced each vector:

```python
from sentence_transformers import SentenceTransformer
import hashlib
import json
import os

MODEL_ID = "all-MiniLM-L6-v2"
STORE_FILE = "versioned_embedding_store.json"

def get_store():
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "r") as f:
            return json.load(f)
    return {"model_id": MODEL_ID, "records": {}}

def embed_and_store(texts, force_recompute=False):
    store = get_store()

    if store["model_id"] != MODEL_ID:
        print(f"WARNING: Store was built with '{store['model_id']}', "
              f"but current model is '{MODEL_ID}'.")
        print("All vectors must be recomputed. Setting force_recompute=True.")
        force_recompute = True
        store = {"model_id": MODEL_ID, "records": {}}

    model = SentenceTransformer(MODEL_ID)

    for text in texts:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in store["records"] and not force_recompute:
            continue
        emb = model.encode([text])[0]
        store["records"][text_hash] = {
            "text": text,
            "model_id": MODEL_ID,
            "vector": emb.tolist(),
        }

    with open(STORE_FILE, "w") as f:
        json.dump(store, f)

    return store

texts = ["lead about pipeline", "technical question about api", "pricing inquiry"]

if os.path.exists(STORE_FILE):
    os.remove(STORE_FILE)

store = embed_and_store(texts)
print(f"Model ID: {store['model_id']}")
print(f"Records stored: {len(store['records'])}")

print("\nSimulating model version mismatch:")
store["model_id"] = "all-MiniLM-L6-v1"
with open(STORE_FILE, "w") as f:
    json.dump(store, f)

store = embed_and_store(["new text after version change"])
print(f"Store rebuilt with model: {store['model_id']}")
print(f"Records after rebuild: {len(store['records'])}")

os.remove(STORE_FILE)
```

The practical takeaway: pin your model version in configuration, store it alongside every vector, and write a migration script that detects version drift and re-embeds your entire corpus atomically. Never mix vectors from different model versions in the same index — the cosine similarities across versions are meaningless.