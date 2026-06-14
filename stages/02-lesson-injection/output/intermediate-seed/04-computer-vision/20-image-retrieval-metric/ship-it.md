## Ship It

Deploying a metric learning retrieval system in a GTM enrichment pipeline means thinking about three operational layers: the embedding service, the index, and the integration into the waterfall.

The **embedding service** is the first bottleneck. Every query image requires a forward pass through your backbone model. On CPU, a ResNet50 takes 50-100ms per image. On GPU, it drops to 2-5ms but you pay for the GPU. For batch processing (enriching a backlog of screenshots), throughput matters more than latency — batch 32-64 images through the model at once. For real-time enrichment (a prospect uploads an asset and you need to respond within seconds), you need the model loaded and warm. A TorchServe or Triton inference server handles both patterns. If your enrichment runs inside a Clay workflow rather than a custom backend, the embedding step becomes an API call — either to a hosted model (OpenAI CLIP via API, Google Vertex AI) or to your own endpoint.

The **index** is where scale decisions matter. FAISS indices live in memory. A flat index for 10,000 logos at 512 dimensions takes about 20MB — trivial. For 1 million product images at 768 dimensions (DINOv2), you need 3GB of RAM and should switch to an IVF index with product quantization to keep memory under 1GB. FAISS indices are not dynamic — you cannot efficiently add new vectors to an IVF index without rebuilding it. If your logo library updates frequently (new companies added to your ICP), you either rebuild the index nightly or use HNSW, which supports incremental additions at the cost of higher memory usage. The tradeoff is: flat index (exact, slow at scale, no rebuild needed) vs. IVF (approximate, fast, periodic rebuild) vs. HNSW (approximate, fast, incremental updates, high memory).

The **enrichment waterfall integration** ties it back to Zone 04. The waterfall pattern is Find → Enrich → Transform → Export. Image retrieval sits in the Enrich step, but it depends on what comes before (Find identifies the prospect and locates their visual assets) and feeds what comes after (Transform turns the logo match into a structured field like "tech_stack": ["Stripe", "Salesforce"]). The Export step sends that enriched record to your CRM or outreach tool. When the retrieval confidence is below threshold, the waterfall should fall through to the next enrichment source — not fail the entire record. A logo match of 0.65 does not block the pipeline; it means "this enrichment source did not produce a confident result, move to the next data provider."

```python
import torch
import torch.nn as nn
import faiss
import numpy as np
import time

class LogoRetrievalEnricher:
    def __init__(self, logo_library, model, preprocess, confidence_threshold=0.85):
        self.model = model
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.labels = [name for name, _, _ in logo_library]

        embeddings = []
        with torch.no_grad():
            for name, img in logo_library:
                emb = model(preprocess(img).unsqueeze(0)).squeeze().numpy().flatten()
                embeddings.append(emb.astype('float32'))

        matrix = np.array(embeddings).astype('float32')
        faiss.normalize_L2(matrix)
        dim = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)

    def enrich(self, query_image):
        with torch.no_grad():
            emb = self.model(self.preprocess(query_image).unsqueeze(0))
            emb = emb.squeeze().numpy().flatten().astype('float32')

        faiss.normalize_L2(emb.reshape(1, -1))
        distances, indices = self.index.search(emb.reshape(1, -1), 3)

        best_idx = indices[0][0]
        best_score = float(distances[0][0])
        best_match = self.labels[best_idx]

        if best_score >= self.confidence_threshold:
            return {
                "enriched": True,
                "company": best_match,
                "confidence": best_score,
                "alternatives": [
                    {"company": self.labels[indices[0][i]], "score": float(distances[0][i])}
                    for i in range(1, 3)
                ],
            }
        return {"enriched": False, "reason": "below_confidence_threshold", "best_guess": best_match, "score": best_score}

def waterfall_enrichment(query_image, enrichers):
    results = []
    for name, enricher in enrichers:
        start = time.time()
        result = enricher.enrich(query_image) if hasattr(enricher, 'enrich') else enricher(query_image)
        elapsed = (time.time() - start) * 1000
        results.append({"source": name, "result": result, "latency_ms": round(elapsed, 1)})

        if isinstance(result, dict) and result.get("enriched"):
            print(f"[{name}] ✓ Enriched: {result['company']} ({result['confidence']:.4f}) in {elapsed:.1f}ms")
            return results
        else:
            print(f"[{name}] ✗ No confident match ({result.get('score', 0):.4f}) in {elapsed:.1f}ms — falling through")

    print(f"[waterfall] All enrichment sources exhausted — no match found")
    return results

model = nn.Sequential(*list(__import__('torchvision').models.resnet50(weights='DEFAULT').children())[:-1])
model.eval()

from PIL import Image
def make_synth(c):
    return Image.fromarray(np.full((224,224,3), c, dtype=np.uint8))

from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

logos = [("Acme Corp", make_synth((220,50,50))), ("Globex", make_synth((50,100,220)))]
logo_enricher = LogoRetrievalEnrichment_Logo = LogoRetrievalEnricher(
    [(n, img) for n, img in logos], model, preprocess, confidence_threshold=0.85
)

domain_lookup = lambda img: {"enriched": False, "reason": "no_domain_found", "score": 0.0}
llm_lookup = lambda img: {"enriched": False, "reason": "low_confidence", "score": 0.6, "best_guess": "Unknown Corp"}

print("=== Enrichment Waterfall: visual asset → company identification ===\n")
query = make_synth((210, 45, 45))
waterfall_enrichment(query, [("logo_retrieval", logo_enricher), ("domain_lookup", domain_lookup), ("llm_vision", llm_lookup)])

print("\n=== Waterfall with low-confidence query (falls through to next source) ===\n")
low_conf_query = make_synth((128, 128, 128))
waterfall_enrichment(low_conf_query, [("logo_retrieval", logo_enricher), ("domain_lookup", domain_lookup), ("llm_vision", llm_lookup)])
```

The first query (a near-match for Acme Corp's red logo) gets enriched at the logo retrieval step with high confidence. The second query (a gray rectangle matching nothing) falls through logo retrieval, falls through domain lookup, and lands at the LLM vision step — which is the last-ditch enrichment source in the waterfall. This is the same Find → Enrich → Transform → Export pattern that Clay implements for structured data enrichment, applied to visual retrieval.

Monitoring in production: track the distribution of cosine similarities over time. If your top-1 similarity scores drift downward, either your index is stale (new logos not yet indexed) or your query distribution has shifted (different image quality from a new data source). Set up alerts for query latency percentiles — if p99 search time exceeds 50ms on a flat index, it is time to switch to IVF or HNSW. And log every below-threshold result: those are the queries that fell through the waterfall, and reviewing them tells you whether your confidence threshold is too strict or your index is missing categories.