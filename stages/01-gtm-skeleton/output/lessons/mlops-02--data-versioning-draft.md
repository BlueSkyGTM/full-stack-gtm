# MLOps 02 — Data Versioning

## Hook It
A model produces different results after retraining. The code didn't change. The hyperparameters didn't change. The data changed — and nobody recorded which version the previous model used. Data versioning is the mechanism that makes training reproducible. Without it, you have artifacts with no provenance.

## See It
Demonstrate content-addressable storage: hash a CSV file with SHA-256, rename it by its hash, and observe that any byte-level change produces a completely different hash. Then show how Git tracks code this way, and how the same pattern applies to arbitrarily large binary files that Git cannot store efficiently. This is the mechanism DVC implements — a content hash layer that sits beside Git.

## Build It
**Easy:** Initialize a DVC project inside a Git repo, add a CSV dataset with `dvc add`, observe the `.dvc` metafile and `.gitignore` entry, commit the metafile to Git. Change one row in the CSV, re-add, commit again. Use `dvc checkout` to restore the previous version. Print the dataset row counts at each version to confirm.

**Medium:** Push the data to local filesystem remote storage (`dvc remote add -d myremote /tmp/dvc-storage`), run `dvc push`, delete the local CSV, and `dvc pull` to restore it. Show the remote directory structure to observe how DVC stores content-addressed files.

**Hard:** Write a script that trains a scikit-learn model on a DVC-tracked dataset, logs the Git commit hash and DVC dataset hash as model metadata, and verifies on load that the current dataset hash matches the logged hash. Print mismatch/success.

## Use It
In GTM Zone 1 (Enrichment), Clay waterfalls pull data from multiple providers, score results, and write enriched records. If the enrichment logic changes or a provider API returns different fields, downstream segmentation breaks. Versioning the enriched dataset with DVC — committing the `.dvc` file alongside the waterfall configuration — creates a reproducible link: "this enrichment config produced this dataset." When a segment underperforms, you can diff the dataset versions and identify which enrichment change caused the shift.

GTM cluster: **Zone 1 — Enrichment waterfall reproducibility** in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`.

## Ship It
Data versioning in production introduces storage cost and pipeline complexity. Each dataset version is a full copy unless the tool supports deduplication (DVC does, via content-addressable storage). CI/CD integration: add a `dvc pull` step before training, fail the pipeline if the dataset hash in the `.dvc` file doesn't match the remote hash. Tag Git commits with model version + dataset version for traceability.

GTM cluster: **Zone 1 — Enrichment pipeline CI/CD**, ensuring that any change to enrichment logic produces a new versioned dataset artifact before segmentation runs.

## Extend It
- **Delta Lake / lakeFS:** Bring Git-style branch and merge semantics to Parquet datasets stored in S3, without copying files. Branch, transform, merge — or rollback if validation fails. [CITATION NEEDED — concept: lakeFS branch-merge workflow for ML datasets]
- **Data lineage:** Track not just which version of a dataset was used, but which upstream sources and transformations produced it. Tools like OpenLineage implement this as a metadata layer.
- **Automated data validation:** Combine data versioning with Great Expectations so that `dvc push` only succeeds if the dataset passes schema and distribution checks.