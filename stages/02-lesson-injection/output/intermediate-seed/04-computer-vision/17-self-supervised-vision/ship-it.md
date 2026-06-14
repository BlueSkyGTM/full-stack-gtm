## Ship It

Deploying self-supervised embeddings into a production enrichment pipeline means embedding thousands of company screenshots in batch, storing them in a vector database, and wiring the similarity query into your enrichment waterfall. The deployment concern that catches teams off guard is embedding drift: if you swap the backbone (e.g., DINOv2 Small → DINOv2 Base), every embedding in your reference database becomes incompatible. You must re-embed the entire corpus or maintain versioned collections.

Here is a batch embedding pipeline that processes a directory of images and exports embeddings for the enrichment waterfall's Transform step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class MockSSLEncoder(nn.Module):
    def __init__(self, output_dim=384):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, output_dim)
    def forward(self, x):
        feats = self.conv(x).squeeze(-1).squeeze(-1)
        return F.normalize(self.proj(feats), dim=1)

torch.manual_seed(42)
encoder = MockSSLEncoder(output_dim=384)
encoder.eval()

def generate_mock_screenshots(company_names, seed_base=42):
    images = {}
    for i, name in enumerate(company_names):
        torch.manual_seed(seed_base + i * 17)
        images[name] = torch.randn(1, 3, 64, 64)
    return images

companies = ["acme_corp", "globex", "initech", "umbrella", "hooli", "pied_piper"]
screenshots = generate_mock_screenshots(companies)

batch = torch.cat(list(screenshots.values()))
with torch.no_grad():
    embeddings = encoder(batch)

embedding_db = {}
for i, name in enumerate(companies):
    embedding_db[name] = embeddings[i].tolist()

output_path = "/tmp/company_visual_embeddings.json"
with open(output_path, "w") as f:
    json.dump({"model_version": "mock_ssl_v1", "embedding_dim": 384, "embeddings": embedding_db}, f)

print(f"Embedded {len(companies)} companies → {output_path}")
print(f"Model version: mock_ssl_v1 | Dim: 384")
print()

with open(output_path) as f:
    loaded = json.load(f)

query_name = "new_prospect"
torch.manual_seed(999)
query_img = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    query_emb = encoder(query_img)[0]

results = []
for name, emb_list in loaded["embeddings"].items():
    ref_emb = torch.tensor(emb_list)
    sim = F.cosine_similarity(query_emb, ref_emb, dim=0).item()
    results.append((name, sim))
results.sort(key=lambda x: -x[1])

print(f"Query: {query_name} → Top matches")
for name, score in results[:3]:
    status = "MATCH" if score > 0.3 else "weak"
    print(f"  {name:15s}: cosine={score:+.4f}  [{status}]")

best_name, best_score = results[0]
print(f"\nEnrichment waterfall decision:")
print(f"  confidence={best_score:+.4f}  →  {'PASS to next waterfall step' if best_score > 0.3 else 'FLAG for manual review'}")
```

In production, the enrichment waterfall calls this embedding step as a custom enrichment provider. The model version string (`mock_ssl_v1`) is critical — when you upgrade the backbone, you version the collection and re-embed. The threshold (0.3 in this mock) should be calibrated on a held-out set of known matches, not guessed. Start by embedding 100-200 known company-logo pairs, computing their cosine similarities, and setting the threshold at the 5th percentile of true-match scores. That gives you a 95% recall floor before you ship.

[CITATION NEEDED — concept: percentage of GTM teams using visual embeddings for firmographic enrichment in production pipelines]