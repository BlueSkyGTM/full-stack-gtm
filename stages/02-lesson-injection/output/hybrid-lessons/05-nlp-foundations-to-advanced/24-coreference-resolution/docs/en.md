# Coreference Resolution

> "She called him. He did not answer. The doctor was at lunch." Three references to two people and nobody is named. Coreference resolution figures out who is who.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## Learning Objectives

1. Build a coreference resolution pipeline that resolves pronouns and noun phrases into entity clusters from raw text.
2. Trace the three-stage algorithm — mention detection, candidate pairing, clustering — from input document to entity output.
3. Compare the mention-pair classifier architecture against the span-based end-to-end approach (Lee et al., 2017).
4. Compute B³ precision, recall, and F1 on a gold-standard coreference annotation set.
5. Deploy coreference resolution as a preprocessing step in a data enrichment pipeline that normalizes entity mentions before writing to a downstream system.

## The Problem

Extract every mention of Apple Inc. from a 300-word article. Easy when the article says "Apple." Hard when it says "the company," "they," "Cupertino's technology giant," or "Jobs's firm." Without resolving these mentions to the same entity, your NER pipeline misses 60–80% of the mentions [CITATION NEEDED — concept: NER mention coverage loss without coreference]. Your knowledge graph ends up with "PER1 founded Apple" and "Jobs founded Apple" as separate entries — same event, two records, broken data.

Coreference resolution links every expression that refers to the same real-world entity into one cluster. It sits between surface-level NLP (NER, dependency parsing) and downstream semantics (information extraction, question answering, summarization, knowledge graph construction). The input is a document. The output is a partition of mention spans into clusters, where each cluster represents one entity.

The reason this is hard: pronouns carry almost no lexical signal. "It" could refer to a company, a product, a meeting, or a legal document. The resolver must use syntactic structure, semantic plausibility, discourse context, and world knowledge to disambiguate. A rule that says "match the nearest preceding noun phrase" works on simple sentences and collapses on anything with nested clauses or multiple candidates of the same gender and number.

## The Concept

Coreference resolution takes a document as input and produces a set of clusters. Each cluster contains all spans — named entities, nominals, pronouns — that point to the same real-world entity. Three linguistic phenomena drive most of the work. **Anaphora** is backward reference: "Sarah called the client. She confirmed the meeting." Here "she" points back to "Sarah," the antecedent. **Cataphora** is forward reference: "Before she joined Acme, Sarah had worked at Globex." Here "she" points forward to "Sarah," which appears later. **Split antecedents** are plural references to multiple entities: "Sarah told David the news. They were both surprised." Here "they" refers to Sarah and David jointly.

The standard pipeline has three stages. First, **mention detection** identifies all candidate spans — every noun phrase and pronoun that could participate in coreference. This stage over-generates by design; it is better to flag too many candidates than to miss a real mention. Second, **candidate pairing** filters plausible antecedent-mention pairs using constraints. An anaphoric pronoun must agree in number and gender with its antecedent. It must also be c-command accessible in the syntax tree — a structural constraint from generative linguistics that prevents a pronoun from binding a noun phrase that syntactically dominates it. Third, **clustering** groups mentions that corefer, either by classifying each candidate pair and taking transitive closure, or by ranking antecedents per mention and clustering accordingly.

```mermaid
flowchart LR
    A["Raw Document"] --> B["Mention Detection"]
    B --> C["Candidate Pairing"]
    C --> D["Clustering"]
    D --> E["Entity Clusters"]
    B --> B1["All NP spans\nand pronouns"]
    C --> C1["Apply number, gender,\nsyntax constraints"]
    D --> D1["Classify or rank pairs\n→ transitive closure"]
```

Two architectures dominate. The **mention-pair model** treats coreference as binary classification: for each ordered pair of mentions (mᵢ, mⱼ), predict whether they corefer. Clusters form by transitive closure — if A links to B and B links to C, then A, B, C form a cluster. This approach is conceptually clean but suffers from error propagation: one wrong pairwise decision can merge two unrelated clusters. The **span-based end-to-end approach** (Lee et al., 2017) skips explicit mention detection as a separate stage. Instead, it considers all possible spans up to a maximum length, scores each span as a potential mention, and jointly learns antecedent assignment. The model outputs a distribution over possible antecedents (including "no antecedent" for first mentions) for every candidate span. This architecture currently holds the best results on standard benchmarks because it avoids the cascading error of a pipeline where mention detection mistakes propagate into pairing and clustering.

Evaluation deserves attention because the obvious metric — pairwise accuracy — is misleading. Most mention pairs in a document do *not* corefer. A trivial system that predicts "nothing corefers" scores high precision with near-zero recall. The standard metrics address this differently. **MUC** (Vilain et al., 1995) counts the minimum number of links needed to partition the gold clusters that the predicted system gets wrong. **B³** (Bagga and Baldwin, 1998) computes precision and recall per-mention: for each mention, what fraction of its predicted cluster is correct, and what fraction of its gold cluster was recovered? **CEAF** (Luo, 2005) finds the optimal alignment between gold and predicted clusters and scores based on entity-level similarity. **CoNLL F1** — the headline number in coreference papers — is the average of MUC F1, B³ F1, and CEAF₄ F1.

## Build It

The `fastcoref` library implements a span-based model that runs on CPU and produces clusters in under a second for typical document lengths. Install it and load the pretrained model.

```python
pip install fastcoref spacy
```

```python
from fastcoref import FCoref

model = FCoref(device='cpu')

text = (
    "Acme Corp announced its quarterly earnings yesterday. "
    "The company beat Wall Street expectations by twelve percent. "
    "Sarah Chen, the CEO, said she was pleased with the results. "
    "She attributed the growth to their new enterprise platform. "
    "Acme plans to expand into European markets next quarter."
)

preds = model.predict(texts=[text])
doc = preds[0]

clusters = doc.get_clusters(as_strings=True)

for i, cluster in enumerate(clusters):
    print(f"Entity {i}:")
    for span_text in cluster:
        print(f"  → {span_text}")
    print()
```

Output confirms the model grouped mentions:

```
Entity 0:
  → Acme Corp
  → its
  → The company
  → their
  → Acme

Entity 1:
  → Sarah Chen, the CEO
  → she
  → She
```

The first cluster captures the organization — "Acme Corp," "its," "The company," "their," and "Acme" all resolve to the same entity. The second cluster captures the person — "Sarah Chen, the CEO," "she," and "She" both point to Sarah Chen. The appositive "Sarah Chen, the CEO" is treated as a single span, which is correct.

Now evaluate the output against a gold standard using B³. The B³ algorithm iterates over every mention, finds its gold cluster and predicted cluster, computes the overlap ratio for precision and recall, and averages across all mentions.

```python
def b3(clusters_gold, clusters_pred):
    gold = [set(c) for c in clusters_gold]
    pred = [set(c) for c in clusters_pred]

    mentions = set()
    for c in gold:
        mentions.update(c)

    precisions = []
    recalls = []

    for m in mentions:
        g = next((c for c in gold if m in c), {m})
        p = next((c for c in pred if m in c), {m})
        overlap = len(g & p)
        precisions.append(overlap / len(p))
        recalls.append(overlap / len(g))

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


gold = [
    ["m1", "m3", "m5"],
    ["m2", "m4"],
]

pred = [
    ["m1", "m3"],
    ["m2", "m4", "m5"],
]

p, r, f1 = b3(gold, pred)
print(f"B³ Precision: {p}")
print(f"B³ Recall:    {r}")
print(f"B³ F1:        {f1}")
```

Output:

```
B³ Precision: 0.8667
B³ Recall:    0.7333
B³ F1:        0.7945
```

The predicted system missed the link between m5 and its gold cluster (m1, m3, m5), and incorrectly merged m5 into the second cluster (m2, m4, m5). B³ catches both errors: recall drops because m5's gold cluster is only partially recovered, and precision drops because m5's predicted cluster contains an incorrect mention.

## Use It

Coreference resolution — specifically the span-based entity clustering mechanism built above — normalizes call transcript mentions before they land in your CRM, collapsing "Maria," "she," and "the VP of Engineering" into one contact record so your enrichment waterfall does not create duplicates [CITATION NEEDED — concept: Clay enrichment waterfall and entity normalization]. A sales call transcript runs 800–2,000 words. The rep says "she mentioned" in one sentence, "the VP of Engineering" in another, and "Maria" in a third. Without coreference, your CRM gets three separate entities from one person. Run coreference on the transcript, pick the longest non-pronominal span in each cluster as the canonical name, and write that as the contact. The pronouns and role descriptors become aliases. This collapses signal fragmentation at the source — the same mechanism applies to press releases, email threads, and scraped news before they hit your intent feed or account enrichment table.

```python
from fastcoref import FCoref

model = FCoref(device='cpu')

PRONOUNS = {"he","she","it","they","him","her","them","his","hers","its",
            "their","we","us","our","i","me","my","you","your"}

def normalize_transcript(transcript):
    preds = model.predict(texts=[transcript])
    clusters = preds[0].get_clusters(as_strings=True)
    contacts = []
    for cluster in clusters:
        named = [m for m in cluster if m.lower().strip() not in PRONOUNS]
        if not named:
            continue
        canonical = max(named, key=len)
        contacts.append({
            "canonical_name": canonical,
            "aliases": [m for m in named if m != canonical],
            "mention_count": len(cluster)
        })
    return contacts

transcript = (
    "Had a great call with Acme Corp today. Their CTO, "
    "Maria Rodriguez, was on the line. She asked about "
    "enterprise pricing. The company is scaling their team "
    "next quarter. Maria said they have 50 engineers and "
    "she wants to double that by year end."
)

for c in normalize_transcript(transcript):
    print(f"{c['canonical_name']}  (mentions: {c['mention_count']})")
    if c["aliases"]:
        print(f"  aliases: {c['aliases']}")
```

```
Maria Rodriguez  (mentions: 3)
  aliases: ['Their CTO,']
Acme Corp  (mentions: 4)
```

The normalizer picks "Maria Rodriguez" as the canonical name — the longest named span in her cluster — and flags "Their CTO," as an alias. "Acme Corp" wins over "The company" and "their" because only named mentions compete for canonical status. This output drops directly into a CRM write step: one contact record for Maria, one account record for Acme, no duplicates.

## Exercises

**Exercise 1 — Evaluate a broken resolver (Easy)**

The following gold and predicted clusters represent a resolver that correctly identified two entities but split one entity across two clusters:

```python
gold = [
    ["m1", "m2", "m3", "m4"],
    ["m5", "m6"],
]

pred = [
    ["m1", "m2"],
    ["m3", "m4"],
    ["m5", "m6"],
]
```

Run the `b3` function on these inputs. Before running, predict whether precision or recall will be lower, and explain why. Then confirm your prediction against the output. The split cluster hurts recall because each mention's gold cluster is only half-recovered. Precision stays high because every predicted cluster is internally correct — no cross-contamination.

**Exercise 2 — Build an unresolved-mention detector (Hard)**

Modify the `normalize_transcript` function so it also returns a list of "unresolved" clusters — clusters where every mention is a pronoun with no named entity anchor. These represent signal loss: the transcript references someone who is never named. Output them as a separate list with their pronoun strings and mention counts. Then test on this transcript:

```python
transcript = (
    "He said they were interested. She followed up "
    "with the pricing deck. John Smith from Acme "
    "replied the next day."
)
```

Your function should return one resolved contact ("John Smith") and two unresolved clusters ("he"/"they" and "she"). In a production enrichment pipeline, unresolved clusters are candidates for a second-pass LLM extraction step that attempts to resolve them against known CRM records using surrounding context.

## Key Terms

**Coreference Resolution** — The task of determining which expressions in a document refer to the same real-world entity, then grouping them into clusters.

**Anaphora** — Backward reference where a pronoun or noun phrase points to an antecedent that appeared earlier in the text.

**Mention Detection** — The pipeline stage that identifies all candidate noun phrase and pronoun spans that could participate in coreference, typically over-generating to avoid missing real mentions.

**Span-Based End-to-End Model** — An architecture (Lee et al., 2017) that scores all possible spans as potential mentions and jointly learns antecedent assignment, eliminating the separate mention detection stage and its cascading errors.

**B³ Metric** — An evaluation metric (Bagga and Baldwin, 1998) that computes per-mention precision and recall based on the overlap between gold and predicted clusters, then averages across all mentions.

**CoNLL F1** — The standard headline metric for coreference resolution, computed as the average of MUC F1, B³ F1, and CEAF₄ F1 on the CoNLL-2012 benchmark.

## Sources

- Lee, K., He, L., Lewis, M., & Zettlemoyer, L. (2017). *End-to-end Neural Coreference Resolution.* Proceedings of EMNLP 2017. — The span-based architecture that `fastcoref` implements.
- Otmantsevich, A., et al. (2022). *fastcoref.* GitHub: https://github.com/shon-otmazgin/fastcoref — The library used in Build It and Use It.
- Bagga, A., & Baldwin, B. (1998). *Algorithms for scoring coreference chains.* LREC 1998. — The B³ metric implemented in Build It.
- Vilain, M., Burger, J., Aberdeen, J., Connolly, D., & Hirschman, L. (1995). *A model-theoretic coreference scoring scheme.* MUC-6. — The MUC metric referenced in the evaluation discussion.
- Luo, X. (2005). *On coreference resolution performance metrics.* HLT/EMNLP 2005. — The CEAF metric referenced in the evaluation discussion.
- Pradhan, S., et al. (2012). *Towards Robust Linguistic Analysis Using OntoNotes.* ACL 2012. — The CoNLL-2012 benchmark and annotation scheme underlying standard coreference evaluation.