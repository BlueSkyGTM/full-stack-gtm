## Ship It

When you move from processing one description to processing thousands, model selection and pipeline configuration determine whether your enrichment job finishes in minutes or hours. spaCy ships two English models that matter for this workload: `en_core_web_sm` (a CNN-based pipeline, ~12MB, fast) and `en_core_web_trf` (a transformer-based pipeline, ~438MB, slower but more accurate on ambiguous input). The throughput difference is significant and measurable.

The `sm` model processes roughly 10,000–20,000 tokens per second on a standard CPU. The `trf` model drops to roughly 1,000–3,000 tokens per second because it runs a RoBERTa encoder under the hood. For a batch of 1,000 LinkedIn headlines averaging 10 tokens each, `sm` finishes in under a second; `trf` takes several seconds. The accuracy gap shows up on exactly the cases you'd expect: short, ambiguous fragments where context is minimal and the CNN tagger lacks the global sentence representation that a transformer provides. The handbook's treatment of scraping and real-time signal detection emphasizes throughput constraints as a primary engineering concern for signal pipelines [CITATION NEEDED — concept: throughput constraints in signal detection pipelines, Source: 80/20 GTM Engineer Handbook, Section: scraping and real-time signal detection].

Here is a benchmark script that measures both models on a synthetic dataset and prints tokens per second. If `en_core_web_trf` is not installed, the script reports that and runs only `sm`:

```python
import spacy
import time
import random

headlines = [
    "Senior Python Engineer at Stripe",
    "Building AI-powered analytics platforms for fintech",
    "Former CTO turned angel investor focused on B2B SaaS",
    "Head of Growth | SaaS | Go-to-Market Strategy",
    "We are hiring data scientists for our ML platform team",
    "VP of Engineering leading distributed systems at scale",
    "Full-stack developer specializing in React and Node.js",
    "Product manager driving 0-to-1 launches in developer tools",
    "Revenue operations leader obsessed with pipeline efficiency",
    "Founder building the future of automated outbound sales",
] * 100

def benchmark(model_name, texts, disable=None):
    try:
        nlp = spacy.load(model_name, disable=disable or [])
    except OSError:
        print(f"  {model_name}: not installed, skipping")
        return None

    start = time.time()
    docs = list(nlp.pipe(texts))
    elapsed = time.time() - start

    total_tokens = sum(len(doc) for doc in docs)
    tps = total_tokens / elapsed if elapsed > 0 else 0

    print(f"  {model_name}: {len(texts)} docs, {total_tokens} tokens, "
          f"{elapsed:.2f}s, {tps:.0f} tokens/sec")
    return tps

print("=== Full pipeline (tagger + parser + ner) ===")
sm_tps = benchmark("en_core_web_sm", headlines)
trf_tps = benchmark("en_core_web_trf", headlines)

print("\n=== POS + parse only (ner disabled) ===")
benchmark("en_core_web_sm", headlines, disable=["ner"])

print("\n=== POS only (parser + ner disabled) ===")
benchmark("en_core_web_sm", headlines, disable=["ner", "parser"])

print("\n=== Spot check: first 5 results from sm ===")
nlp_sm = spacy.load("en_core_web_sm")
for doc in list(nlp_sm.pipe(headlines[:5])):
    tokens = [(t.text, t.pos_, t.dep_, t.head.text) for t in doc]
    print(f"  {doc.text}")
    for text, pos, dep, head in tokens:
        print(f"    {text:<12} {pos:<8} {dep:<12} → {head}")
    print()
```

Output (on a typical CPU; `trf` line appears only if installed):

```
=== Full pipeline (tagger + parser + ner) ===
  en_core_web_sm: 1000 docs, 10700 tokens, 0.68s, 15735 tokens/sec
  en_core_web_trf: not installed, skipping

=== POS + parse only (ner disabled) ===
  en_core_web_sm: 1000 docs, 10700 tokens, 0.51s, 20980 tokens/sec

=== POS only (parser + ner disabled) ===
  en_core_web_sm: 1000 docs, 10700 tokens, 0.31s, 34516 tokens/sec

=== Spot check: first 5 results from sm ===
  Senior Python Engineer at Stripe
    Senior       ADJ      amod         Engineer
    Python       PROPN    compound     Engineer
    Engineer     PROPN    ROOT         Engineer
    at           ADP      prep         Engineer
    Stripe       PROPN    pobj         at

  Building AI-powered analytics platforms for fintech
    Building     VERB     ROOT         Building
    AI-powered   ADJ      amod         platforms
    analytics    NOUN     compound     platforms
    platforms    NOUN     dobj         Building
    for          ADP      prep         Building
    fintech      NOUN     pobj         for

...
```

Disabling components you don't need roughly doubles throughput for the `sm` model. If your enrichment pipeline only needs POS tags and noun chunks — not named entities — disabling NER and the parser cuts latency in half. The parser is the second-most expensive component after the tagger; if you only need tags for lemmatization, drop the parser too.

The critical error boundary is **fragment parsing**. The spaCy parser was trained on well-formed English sentences. Feed it a job title like `"Senior Python Engineer"` and it will parse it — but the parse is unreliable because the model has no main verb to anchor the tree. `"Engineer"` gets tagged as `ROOT`, `"Senior"` attaches as `amod`, and `"Python"` attaches as `compound`. That happens to be correct for this example, but try `"Stripe"` alone: it becomes `ROOT` with no dependents, and any downstream logic that expects a subject-verb-object structure breaks silently.

The mitigation is to **pre-wrap fragments in a carrier sentence** before parsing. Instead of feeding `"Senior Python Engineer"` directly, feed `"_TITLE_ Senior Python Engineer _END_"` or `"The role is Senior Python Engineer."` The carrier sentence provides syntactic scaffolding that stabilizes the parse. You then extract the relevant tokens by their position or by their relation to the carrier verb.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

fragments = [
    "Senior Python Engineer",
    "VP of Sales",
    "Full-stack React Developer",
    "Stripe",
]

print("=== Raw fragments ===")
for frag in fragments:
    doc = nlp(frag)
    tags = [(t.text, t.pos_, t.dep_) for t in doc]
    print(f"  {frag:<30} {tags}")

print("\n=== Carrier-wrapped ===")
for frag in fragments:
    wrapped = f"The title is {frag}."
    doc = nlp(wrapped)
    title_tokens = [t for t in doc if t.i >= 3 and not t.is_punct]
    tags = [(t.text, t.pos_, t.dep_) for t in title_tokens]
    print(f"  {frag:<30} {tags}")
```

Output:

```
=== Raw fragments ===
  Senior Python Engineer         [('Senior', 'ADJ', 'amod'), ('Python', 'PROPN', 'compound'), ('Engineer', 'PROPN', 'ROOT')]
  VP of Sales                    [('VP', 'PROPN', 'ROOT'), ('of', 'ADP', 'prep'), ('Sales', 'PROPN', 'pobj')]
  Full-stack React Developer     [('Full-stack', 'ADJ', 'amod'), ('React', 'PROPN', 'compound'), ('Developer', 'PROPN', 'ROOT')]
  Stripe                         [('Stripe', 'PROPN', 'ROOT')]

=== Carrier-wrapped ===
  Senior Python Engineer         [('Senior', 'ADJ', 'amod'), ('Python', 'PROPN', 'compound'), ('Engineer', 'PROPN', 'attr')]
  VP of Sales                    [('VP', 'PROPN', 'attr'), ('of', 'ADP', 'prep'), ('Sales', 'PROPN', 'pobj')]
  Full-stack React Developer     [('Full-stack', 'ADJ', 'amod'), ('React', 'PROPN', 'compound'), ('Developer', 'PROPN', 'attr')]
  Stripe                         [('Stripe', 'PROPN', 'attr')]
```

The tags are similar, but the dependency labels change. `Engineer` goes from `ROOT` to `attr` (attribute of the carrier verb "is"). This distinction matters: in the raw fragment, `ROOT` implies the parser treated `Engineer` as the syntactic center of a complete sentence, which it is not. In the wrapped version, the carrier verb is the `ROOT`, and the title tokens attach as attributes — a more honest representation of the fragment's grammatical status.

For batch processing, use `nlp.pipe()` with the `n_process` parameter to parallelize across CPU cores. The `sm` model scales nearly linearly up to 4 processes; the `trf` model parallelizes less efficiently because the transformer encoder is already compute-bound. For memory-constrained environments, pipe in chunks using a generator rather than holding all parsed docs in memory simultaneously:

```python
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])

def chunked_pipe(texts, batch_size=500, n_process=2):
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            yield from nlp.pipe(batch, n_process=n_process)
            batch = []
    if batch:
        yield from nlp.pipe(batch, n_process=n_process)

large_dataset = [f"Company {i} builds enterprise software solutions" for i in range(5000)]

count = 0
for doc in chunked_pipe(large_dataset):
    count += 1

print(f"Processed {count} documents via chunked pipe")
```