## Ship It

Shipping a RAG system into a GTM workflow means committing to a qrels fixture that represents real queries your reps ask. Build it from actual questions logged in your sales enablement tool or Slack. For each query, label which case studies, docs, or pricing pages are genuinely relevant. This is manual work — there is no shortcut. A qrels file with 50 well-labeled queries is more useful than 500 auto-generated ones.

```python
import json

qrels_fixture = {
    "queries": [
        {
            "id": "q1",
            "text": "Which customers reduced onboarding time?",
            "relevant_doc_ids": ["d1"],
            "gold_answer": "Acme Corp reduced onboarding time by 40 percent."
        },
        {
            "id": "q2",
            "text": "What is the pricing for enterprise plans?",
            "relevant_doc_ids": ["d8"],
            "gold_answer": "Enterprise plans start at 50K per year with custom SLAs."
        },
        {
            "id": "q3",
            "text": "Do we have case studies for agent frameworks?",
            "relevant_doc_ids": ["d5", "d7"],
            "gold_answer": "Umbrella Corp deployed our agent framework across 500 support seats. Wayne Enterprises integrated our RAG pipeline."
        }
    ],
    "corpus": corpus
}

with open("qrels.json", "w") as f:
    json.dump(qrels_fixture, f, indent=2)

print("Wrote qrels.json with", len(qrels_fixture["queries"]), "queries and",
      len(qrels_fixture["corpus"]), "documents")
print()

with open("qrels.json") as f:
    loaded = json.load(f)

for q in loaded["queries"]:
    print(f"  {q['id']}: {q['text']}")
    print(f"    relevant: {q['relevant_doc_ids']}")
```

Output:

```
Wrote qrels.json with 3 queries and 8 documents

  q1: Which customers reduced onboarding time?
    relevant: ['d1']
  q2: What is the pricing for enterprise plans?
    relevant: ['d8']
  q3: Do we have case studies for agent frameworks?
    relevant: ['d5', 'd7']
```

In production, the eval loop looks like this. You have two retrieval configurations — say, your current chunking strategy (500-token chunks with 50-token overlap) and a candidate (250-token chunks with 100-token overlap). Run both against the qrels fixture. Compare the mean precision, recall, MRR, and nDCG. If the candidate improves recall by 15% without tanking precision, ship it. If faithfulness drops, the smaller chunks are losing context the generator needs — do not ship it.

The same comparison applies to reranker changes, embedding model swaps, and prompt changes. Each change is an A/B test against the qrels fixture. The metrics tell you which configuration to deploy. Without this loop, you are changing one variable at a time and hoping the outputs look better — which is not engineering, it is gambling.

When you wire the metrics into a CI pipeline, the goal is a gate: if mean faithfulness drops below 0.90 or mean answer relevance drops below 0.60 on the fixture, the deploy fails. RAGAS and DeepEval both expose pytest-style assertions for this. The fixture is version-controlled alongside the code, and every PR that touches the retriever, chunker, or prompt template runs the eval automatically.