## Ship It

Deploy coreference resolution as a preprocessing step in a data enrichment pipeline. The pipeline accepts raw text — transcript, email body, scraped news — runs coreference, and outputs a normalized entity map with mention counts and canonical names.

```python
from fastcoref import FCoref
import json

model = FCoref(device='cpu')

def resolve_entities(text):
    preds = model.predict(texts=[text])
    doc = preds[0]
    clusters = doc.get_clusters(as_strings=True)

    entity_map = {}
    for i, cluster in enumerate(clusters):
        sorted_mentions = sorted(cluster, key=len, reverse=True)
        canonical = sorted_mentions[0] if sorted_mentions else f"entity_{i}"

        def is_pronoun(s):
            return s.lower().strip() in {
                "he", "she", "it", "they", "him", "her",
                "them", "his", "hers", "its", "their", "theirs",
                "we", "us", "our", "i", "me", "my", "you", "your"
            }

        named_mentions = [m for m in cluster if not is_pronoun(m)]
        if named_mentions:
            canonical = sorted(named_mentions, key=len, reverse=True)[0]

        entity_map[f"entity_{i}"] = {
            "canonical_name": canonical,
            "all_mentions": list(cluster),
            "mention_count": len(cluster),
            "has_named_mention": len(named_mentions) > 0
        }

    return entity_map


transcript = (
    "Had a great call with Acme Corp today. Their CTO, "
    "Maria Rodriguez, was on the line. She asked about "
    "our pricing tier for enterprise. The company is "
    "looking to scale their engineering team next quarter. "
    "Maria mentioned they currently have 50 engineers and "
    "she wants to double that. She'll be the