# Data Pipelines for Pre-Training

---

## Beat 1: Hook

The model is the data. Every behavior your LLM exhibits — hallucination rate, domain fluency, instruction-following precision — traces back to what went into the training corpus. This lesson covers the pipeline that moves raw text from source to tokenized, deduplicated shards ready for pre-training.

---

## Beat 2: Concept

**Extraction** pulls raw text from heterogeneous sources (web crawls, PDFs, code repos). **Filtering** applies heuristic and classifier-based quality gates to discard garbage. **Deduplication** removes exact and fuzzy duplicates using MinHash and suffix-array methods. **Tokenization** converts cleaned text into integer sequences a model can consume. These four stages form a linear DAG — each stage's output is the next stage's input.

Key mechanisms: document-level vs. paragraph-level filtering, Bloom filters for near-deduplication, and why deduplication order matters (filter first, dedupe second to avoid paying for deduplication on content you'll discard).

---

## Beat 3: Setup

**Exercise hook (easy):** Install `datasets`, `tokenizers`, and `datasketch`. Download a 10,000-row sample of the FineWeb-Edu dataset from HuggingFace. Print the schema and confirm field names.

**Exercise hook (medium):** Write a function that loads the sample into a streaming `IterableDataset` and prints the first three documents with their metadata fields. Observe memory usage — streaming should hold constant memory regardless of corpus size.

---

## Beat 4: Build

**Exercise hook (hard):** Build a complete pipeline script that: (1) streams documents from the FineWeb-Edu sample, (2) applies a length filter (discard documents under 50 characters and over 100,000 characters), (3) computes MinHash signatures on 5-gram shingles and discards documents exceeding 0.8 Jaccard similarity to any previously seen document, (4) tokenizes surviving documents with a GPT-2 BPE tokenizer, and (5) writes tokenized shards to disk in `.jsonl` format. Print counts at each stage: `extracted → filtered → deduped → tokenized → written`.

---

## Beat 5: Use It

This pipeline pattern — sequential stages that each narrow and transform the dataset — is structurally identical to the **Enrichment Waterfall** in GTM (Zone 04, cluster: "Data pipelines, ETL"). In Clay, the waterfall is: **Find → Enrich → Transform → Export**. You find prospects, enrich with firmographic data, transform by scoring or filtering, and export to your CRM. The same DAG thinking applies: stage order matters, each step reduces downstream volume, and intermediate outputs must be inspectable. The filtering logic you wrote for pre-training data quality maps directly to lead scoring logic — both are classifier-in-the-loop gates on a flowing stream.

**Exercise hook (medium):** Take the pipeline from Beat 4 and refactor it into four named stages: `find`, `enrich`, `transform`, `export`. Each stage is a function that takes a list of dicts and returns a list of dicts. Add a `stage_counts` dictionary that tracks record volume entering and exiting each stage. Run the pipeline and print the flow report.

---

## Beat 6: Ship It

**Exercise hook (hard):** Productionalize the Beat 5 pipeline into a CLI tool (`pipeline.py`) that accepts a YAML config file specifying: source dataset name, filter thresholds, deduplication similarity threshold, tokenizer name, output path, and shard size. The script reads the config, runs the full waterfall, writes sharded output, and prints a final summary: total input records, records surviving each stage, output token count, and disk size. Run it with two different configs (aggressive filtering vs. permissive) and compare the output reports side by side.

---

## Learning Objectives

1. **Implement** a four-stage data pipeline (extract, filter, dedupe, tokenize) that processes a real corpus from raw download to training-ready shards.
2. **Configure** MinHash-based near-deduplication with a tunable Jaccard similarity threshold and measure its effect on corpus size.
3. **Compare** the pre-training data pipeline DAG to the GTM enrichment waterfall pattern, identifying structural correspondences between stages.
4. **Evaluate** the impact of filter threshold choices on output dataset volume and quality by running the pipeline under two configurations and interpreting the resulting statistics.

---

## GTM Redirect Rules (for this lesson)

- **Use It** (Beat 5): Names GTM Zone 04 cluster "Enrichment Waterfalls (1.2 tooling)" explicitly. States: "This is the Clay waterfall — Find → Enrich → Transform → Export." The structural analogy is exact: both are linear DAGs with classifier-in-the-loop gates.
- **Ship It** (Beat 6): The refactored pipeline uses waterfall stage names directly. The config-driven approach mirrors how enrichment workflows are parameterized in production GTM tooling.