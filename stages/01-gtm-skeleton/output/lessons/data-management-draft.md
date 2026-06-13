# Data Management

## Beat 1: Hook

Data is the input to every model, every enrichment, every signal. Bad data architecture produces silent failures—duplicate records, schema drift, null cascades—that compound in production. This lesson covers how to structure, validate, and transform data so downstream systems receive clean, predictable inputs.

---

## Beat 2: Concept

**Core mechanisms to cover:**

- **Schema contracts**: Define what a record must contain before it enters your pipeline. A schema is a enforcement boundary—anything that passes is guaranteed to have required fields with expected types.
- **Data normalization**: Converting heterogeneous inputs (API responses, CSV uploads, scraped HTML) into a single canonical shape. This is the step that prevents downstream `if/else` sprawl.
- **Validation as a gate**: Data passes through a validation function before being written. Invalid records are rejected or quarantined, not silently stored.
- **Immutability and append-only logs**: Writes add records rather than overwriting. This preserves history and enables replay.
- **Deduplication strategies**: Hash-based (compute fingerprint, skip if seen) vs. merge-based (keep most recent, merge fields).

**Key terms to define:** schema, canonical form, append-only, idempotency, deduplication key, quarantine.

---

## Beat 3: Demo

**Working code examples (all run in terminal, print observable output):**

1. **Schema validation function** — Define a schema as a dict, validate incoming records against it, print pass/fail per record.
2. **Normalization pipeline** — Take 3 different input shapes (snake_case API response, CamelCase CRM export, messy CSV row), normalize to one canonical shape, print before/after.
3. **Append-only log with deduplication** — Write records to a JSONL file, compute hash of dedup key, skip if hash already seen, print "written" vs. "skipped" counts.
4. **Quarantine mechanism** — Split a batch into valid and invalid buckets based on schema, write valid to main log, write invalid to quarantine file, print counts for each.

No comments in code. Every example prints output confirming the mechanism worked.

---

## Beat 4: Use It

**GTM Redirect: Zone 03 — Data Enrichment / Waterfall Operations**

Every enrichment waterfall in Clay (or any enrichment tool) depends on clean, normalized input records. If company name is missing or domain format is inconsistent, the waterfall silently returns empty results. The schema validation and normalization patterns from this lesson are what prevent that failure mode.

**Specific applications:**
- Schema contracts map to Clay column requirements (a waterfall on `domain` fails if the field contains `acme.com/about` instead of `acme.com`)
- Deduplication maps to preventing duplicate enrichment runs on the same lead
- Normalization maps to standardizing data across multiple enrichment sources before writing back to CRM
- Quarantine maps to routing unmappable records to a manual review list instead of dropping them

**Exercise hooks:**
- (Easy) Write a schema validator for a GTM lead record (fields: email, company_domain, title). Test against 5 sample records, 2 intentionally invalid.
- (Medium) Build a normalization function that accepts CRM export data (CamelCase, mixed null formats) and returns canonical snake_case with explicit `None` values. Print transformation summary.
- (Hard) Implement an append-only JSONL enrichment log with hash-based dedup on `(email, domain)`. Simulate a duplicate enrichment run and confirm the second run produces zero new writes.

---

## Beat 5: Ship It

**Production deployment considerations:**

- **File format selection**: JSONL for append-only logs (one record per line, crash-safe). Parquet for analytical queries. CSV only for human export, never for internal pipelines.
- **Storage location**: Local filesystem for single-machine scripts. S3/GCS for multi-process or scheduled jobs. SQLite for structured querying of intermediate data.
- **Backup and recovery**: Append-only logs are recovery-friendly by default—reprocess from timestamp. Overwrite-based storage requires explicit backup.
- **Monitoring**: Track record counts per batch (valid, invalid, duplicate). Alert if invalid rate exceeds threshold. [CITATION NEEDED — concept: data quality monitoring alerting thresholds in GTM pipelines]

**Exercise hooks:**
- (Easy) Convert the demo JSONL append log to write to a specified directory with date-stamped filenames. Confirm files exist and contain expected record counts.
- (Medium) Add a summary report function that reads a JSONL log and prints: total records, unique records, duplicates skipped, quarantine count. Run against a log with 100+ records.
- (Hard) Build a SQLite-backed data store with the same schema validation and dedup logic. Compare write performance against JSONL for 10,000 records. Print timing results.

---

## Beat 6: Evaluate

**Assessment hooks (grounded in lesson objectives):**

- Write a schema that enforces: `email` (string, contains `@`), `domain` (string, no trailing slash), `source` (enum: one of 3 allowed values). Validate a provided list of 10 records and output the rejection reasons.
- Given two different API response shapes (provided as code), write a normalization function that produces identical output for both. Print side-by-side to confirm.
- Implement a dedup function using content hashing. Given a dataset with 5 intentional duplicates, confirm exactly 5 are skipped. Print the duplicate keys found.
- Explain in 2–3 sentences why append-only storage is preferred over overwrite for data pipelines that feed enrichment waterfalls. Reference specific failure modes.

---

## Learning Objectives (3–5, action verbs only)

1. **Implement** schema validation functions that reject records missing required fields or containing wrong types.
2. **Normalize** heterogeneous input data into a single canonical shape using explicit mapping logic.
3. **Build** append-only data logs with hash-based deduplication to prevent duplicate processing.
4. **Configure** quarantine paths that separate invalid records from the main data pipeline.
5. **Compare** storage formats (JSONL, SQLite, CSV) based on append-safety, query capability, and crash recovery.