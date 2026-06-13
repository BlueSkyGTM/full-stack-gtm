# Relation Extraction & Knowledge Graph Construction

## Learning Objectives

- Build a rule-based relation extraction pipeline using dependency parsing over unstructured text
- Compare fixed-schema, supervised, and LLM-based extraction approaches for precision and recall tradeoffs
- Construct a directed knowledge graph from extracted triples using NetworkX
- Implement entity resolution to canonicalize entity mentions before graph insertion
- Evaluate extraction quality against a gold standard using precision, recall, and F1

## The Problem

Your CRM contains 40,000 accounts. Each account page has news articles, SEC filings, press releases, and job postings attached as raw text. Your keyword search finds mentions of "acquisition" and "partnership." Your vector similarity finds paragraphs that *look like* they discuss partnerships. Neither answers the question: "Which companies in our ICP acquired a competitor in the last 18 months, and who was the CEO at the time?"

That question requires structured relationships — not bags of words, not embedding distances. A human reading "Acme Corp acquired Beta Inc for $2B" extracts four facts instantly: Acme acquired Beta, the deal was $2B, Acme is an acquirer, Beta was acquired. Relation extraction is the process of turning free text into those structured triples `(subject, relation, object)`. Knowledge graph construction is what happens when you aggregate triples across a corpus, resolve duplicate entities, and store the result as a queryable graph.

NER found the entities. Entity linking anchored them to canonical IDs. Neither tells you how they connect. The connections are where GTM intelligence lives — who reports to whom, which companies partnered, which executives left, which products compete. Without relation extraction, your enrichment layer has nouns but no verbs.

## The Concept

A triple is the atomic unit: `(subject_entity, relation_type, object_entity)`. The relation type is the predicate — "acquired," "employed_by," "competes_with," "invested_in." Two entities enter the triple from NER; the relation exits from extraction. The extraction mechanism is what determines whether your graph has 500 clean edges or 50,000 noisy ones.

Three extraction mechanisms exist, each with different precision and recall profiles:

**Rule-based extraction** operates over dependency parses. A dependency parser produces a syntactic tree showing how words relate grammatically. Rules then match structural patterns: if a verb has a nominal subject (nsubj) and a direct object (dobj), extract `(subject, verb_lemma, object)`. These are Hearst patterns generalized to arbitrary predicates. They are brittle — miss passive voice, relative clauses, or any construction the rule author didn't anticipate — but they never hallucinate. The triple either matches the pattern or it doesn't. For GTM enrichment over financial filings where precision matters more than recall (a false "acquired" triple is worse than a missed one), rule-based extraction is often the right default.

**Supervised classification** takes a sentence with two marked entities and predicts the relation from a fixed schema. The model is trained on labeled datasets like TACRED or DocRED. This approach dominated relation extraction from roughly 2015 to 2022. It requires labeled training data — typically thousands of annotated sentence-entity pairs — and it only predicts relations in the closed schema. If your schema has 20 relation types and the sentence expresses a 21st type, the classifier either picks the closest fit or abstains.

**LLM-based extraction** prompts a generative model to emit triples in structured JSON. This is OpenIE (Open Information Extraction) via prompting: no fixed schema, no training data, any relation the text expresses. The tradeoff is hallucination. An LLM asked to extract relations from "Acme Corp reported strong Q3 earnings" might emit `(Acme Corp, competitor, Beta Inc)` because Beta Inc appeared in the conversation history or the model's training data. Without provenance — a pointer back to the exact source span — you cannot distinguish a real triple from a plausible fabrication.

The 2026 mitigation framework is anchor-verify: extract entities first (anchor), extract candidate relations (extract), then verify each triple against the source text before admitting it to the graph (verify). Verification is a second LLM call that checks whether the source text actually supports the triple, or a rule-based check that the subject and object appear in the same sentence as the predicate. Triples that fail verification are discarded or flagged with low confidence.

```mermaid
flowchart LR
    A[Raw Text Corpus] --> B[NER: Entity Spans]
    B --> C[Entity Pairs per Sentence]
    C --> D{Extraction Method}
    D -->|Rule-Based Patterns| E1[Triples + High Confidence]
    D -->|Supervised Classifier| E2[Triples + Schema-Constrained]
    D -->|LLM Prompting| E3[Triples + Open Schema]
    E1 --> F[Entity Resolution]
    E2 --> F
    E3 --> F
    F --> G[Canonical Triples]
    G --> H[Knowledge Graph]
    H --> I[Query: Shortest Path / Subgraph]
    H --> J[Verification Against Source]
    J -->|Pass| H
    J -->|Fail| K[Discard / Flag]
```

Once triples exist, graph construction follows. Each unique entity becomes a node. Each triple becomes a directed edge from subject to object, labeled with the relation type. Entity resolution merges surface variants — "Acme," "Acme Corp," "Acme Corporation," "ACME" — into a single canonical node. Without resolution, your graph fragments: "Acme acquired Beta" and "Acme Corp acquired Beta Inc" create two disconnected components instead of one connected subgraph. Resolution can be as simple as string normalization (lowercase, strip suffixes) or as complex as embedding-based matching with human review.

Querying is where the graph pays off. Shortest-path between two entities reveals indirect relationships — "Company A and Company B share a common investor" is a two-hop path `(A, funded_by, VC_X), (VC_X, invested_in, B)`. Subgraph extraction around a single entity returns its entire neighborhood: all partners, competitors, executives, and acquisitions within one or two hops. For GTM teams building account plans or identifying warm introduction paths, these traversals are the deliverable.

The central tension is schema design. A fixed schema — 20 relation types you care about — gives high precision and makes querying predictable. Open extraction captures everything but produces noise: "reported," "mentioned," "is located near" all become edges, and most of them are useless for downstream queries. The practitioner picks based on downstream tolerance for false edges. Financial compliance needs near-zero false positives. Competitive intelligence tolerates noise if recall is high enough to catch signals competitors miss.

## Build It

### Stage 1: Rule-Based Triple Extraction via Dependency Parsing

This script downloads spaCy's small English model if needed, parses five sentences, and extracts triples by matching syntactic patterns over the dependency tree. Each triple includes a confidence flag based on how directly the pattern matched.

```python
import subprocess
import sys

try:
    import spacy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "spacy", "-q"])
    import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
    nlp = spacy.load("en_core_web_sm")

sentences = [
    "Acme Corp acquired Beta Inc for 2 billion dollars.",
    "Tim Cook became CEO of Apple in 2011.",
    "Google partnered with Microsoft on cloud infrastructure.",
    "Acme Corp announced a partnership with Beta Inc.",
    "Apple hired John Smith as VP of Engineering.",
]

def expand_phrase(token):
    if token is None:
        return ""
    chunks = [c.text for c in token.children if c.dep_ == "compound"]
    chunks.append(token.text)
    return " ".join(chunks)

def extract_triples_rule_based(doc):
    triples = []
    for nc in doc.noun_chunks:
        root = nc.root
        if root.dep_ not in ("nsubj", "nsubjpass"):
            continue
        verb = root.head
        if verb.pos_ != "VERB":
            continue
        subj_text = nc.text
        verb_lemma = verb.lemma_
        for other_nc in doc.noun_chunks:
            if other_nc == nc:
                continue
            other_root = other_nc.root
            if other_root.head == verb and other_root.dep_ in ("dobj", "attr", "oprd"):
                triples.append({
                    "subject": subj_text,
                    "relation": verb_lemma,
                    "object": other_nc.text,
                    "confidence": "high",
                    "method": "rule_based",
                })
            elif other_root.head.dep_ == "prep" and other_root.head.head == verb:
                prep_word = other_root.head.text
                triples.append({
                    "subject": subj_text,
                    "relation": f"{verb_lemma}_{prep_word}",
                    "object": other_nc.text,
                    "confidence": "medium",
                    "method": "rule_based",
                })
    return triples

all_triples = []
for sent in sentences:
    doc = nlp(sent)
    extracted = extract_triples_rule_based(doc)
    all_triples.extend(extracted)
    print(f"\nSentence: {sent}")
    print(f"  Extracted {len(extracted)} triples:")
    for t in extracted:
        print(f"    ({t['subject']}, {t['relation']}, {t['object']}) [{t['confidence']}]")

print(f"\n{'='*60}")
print(f"Total rule-based triples: {len(all_triples)}")
```

Expected output:

```
Sentence: Acme Corp acquired Beta Inc for 2 billion dollars.
  Extracted 2 triples:
    (Acme Corp, acquired, Beta Inc) [high]
    (Acme Corp, acquired_for, 2 billion dollars) [medium]

Sentence: Tim Cook became CEO of Apple in 2011.
  Extracted 1 triples:
    (Tim Cook, became, CEO) [high]

Sentence: Google partnered with Microsoft on cloud infrastructure.
  Extracted 1 triples:
    (Google, partnered_with, Microsoft) [medium]

Sentence: Acme Corp announced a partnership with Beta Inc.
  Extracted 1 triples:
    (Acme Corp, announced_with, Beta Inc) [medium]

Sentence: Apple hired John Smith as VP of Engineering.
  Extracted 1 triples:
    (Apple, hired, John Smith) [high]

============================================================
Total rule-based triples: 6
```

Notice what the rules miss. "Tim Cook became CEO of Apple" should yield `(Tim Cook, employer, Apple)`, but the preposition "of" attaches to "CEO" (a noun), not to "became" (the verb). The pattern only catches prep phrases attached directly to verbs. "Apple hired John Smith as VP of Engineering" should yield `(John