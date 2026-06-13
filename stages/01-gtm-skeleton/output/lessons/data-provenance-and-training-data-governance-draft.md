# Data Provenance and Training-Data Governance

## Hook

You trained a classifier on 50k labeled support tickets. Six months later, someone asks: "Which tickets came from the EMEA region, were they consented for ML use, and did the label schema change between batch 3 and batch 4?" If you can't answer in under five minutes, you have a provenance problem.

## Concept

**Data provenance** is the chain-of-custody record for every row that enters a training pipeline: origin, transformation history, consent status, and license. **Training-data governance** is the policy layer that decides what provenance must be tracked and who approves changes.

Mechanisms to cover:
- **Lineage graphs**: DAGs that map source → transform → dataset → model. Each edge is a versioned, timestamped operation.
- **Data cards / datasheets**: Structured metadata documents (origin, collection method, labeling schema, known biases, license). Originally from Gebru et al.'s "Datasheets for Datasets" [CITATION NEEDED — concept: Datasheets for Datasets paper and citation].
- **Content origin signals**: C2PA and similar standards embed cryptographic provenance in media files. Relevant when training on images or documents where authenticity matters.
- **Contamination detection**: Identifying when evaluation data has leaked into training data through shared sources or overlapping crawls.
- **Hash-based deduplication and verification**: Using SHA/MinHash to detect duplicate or near-duplicate records across dataset versions.

Tools that implement these mechanisms:
- **DVC (Data Version Control)**: Git-like versioning for datasets and models. Tracks lineage via `dvc.yaml` pipelines.
- **W&B Artifacts**: Weights & Biases tracks dataset versions, model versions, and the edges between them in a graph.
- **Great Expectations**: Validates schema and statistical properties at ingest. Catches schema drift before it poisons training.

## Demonstration

Three runnable examples:

1. **Hash-based provenance tracking** (Python only). Compute SHA-256 hashes for each record in a synthetic dataset. Detect which records changed between two dataset versions by comparing hash sets. Print diff output.

2. **Schema validation at ingest** (Python only). Build a minimal schema validator that checks field presence, types, and allowed values for incoming records. Demonstrate catching a schema violation. Print pass/fail output.

3. **Lineage DAG construction** (Python only). Build a small lineage graph using dictionaries: each node is a dataset version, each edge records the transform applied and timestamp. Traverse the graph to answer "what transforms produced dataset_v3?" Print the traversal path.

## Guided Exercise

**Easy**: Extend the hash-based provenance tracker to also record the source file and timestamp for each record hash. Output a provenance report.

**Medium**: Build a contamination detector. Given two CSV strings (training set, eval set), compute MinHash signatures for each row using a sliding shingle approach. Flag any row in eval that has Jaccard similarity > 0.8 with any training row. Print flagged pairs.

**Hard**: Implement a minimal data card generator. Given a dataset (list of dicts) and a YAML config specifying required metadata fields, validate that all metadata is present, compute summary statistics (class distribution, null rates, date range), and output a Markdown data card document.

## Use It

In GTM enrichment workflows, data provenance determines whether you can legally and operationally use a third-party contact list, intent signal, or technographic dataset. This maps to **Zone 1 (Data Foundation)** in the GTM topic map — specifically the enrichment and data sourcing cluster.

Concrete application: when you pull company data from an enrichment provider into Clay or a similar tool, provenance tracking answers: "Which provider did this field come from? When was it last refreshed? What's the license constraint on storing vs. reselling it?" Without this, enrichment pipelines become untraceable, and compliance reviews become archaeological digs.

Exercise hook: given a synthetic enrichment log with provider, field, timestamp, and license columns, write a query that identifies all fields sourced from a provider whose license expired before the field was last used in an outreach sequence. Print the non-compliant records.

## Ship It

Build a standalone provenance tracker that:
1. Accepts a dataset (JSON array of objects) and a metadata dict (source, license, collection date, consent status)
2. Computes per-row hashes
3. Validates schema against a provided spec
4. Stores everything in a SQLite database with tables for `datasets`, `records`, and `lineage_edges`
5. Exposes a CLI that can answer: "show me the full lineage for dataset X" and "flag any records in dataset Y that also appear in dataset Z"

No frameworks beyond Python stdlib + sqlite3. Print all outputs to terminal. This is a minimal but real governance tool you could extend for production use.

---

**Learning Objectives** (for reference — to be placed in `docs/en.md`):
1. Implement hash-based record fingerprinting to detect data changes across dataset versions.
2. Construct a lineage DAG that records transform history for any dataset version.
3. Validate incoming data against a schema specification and report violations.
4. Detect data contamination between training and evaluation sets using similarity hashing.
5. Generate a structured data card with provenance metadata and summary statistics.