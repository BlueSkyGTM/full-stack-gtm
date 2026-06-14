## Ship It

To ship a data pipeline into production, three things must be true: the pipeline is reproducible, the data is versioned, and the schemas are enforced at every boundary.

**Reproducible splits.** When creating train/validation/test splits for any model component in your pipeline, fix the random seed so the same input produces the same split every time. Without this, you cannot tell whether a performance change is due to your code or due to a different random partition.

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
split = dataset.train_test_split(test_size=0.2, seed=42)

train = split["train"]
test = split["test"]
print(f"Train: {len(train)} | Test: {len(test)}")
print(f"Train[0] label: {train[0]['label']}")
print(f"Test[0] label: {test[0]['label']}")

repeat_split = dataset.train_test_split(test_size=0.2, seed=42)
print(f"Repeat train[0] label: {repeat_split['train'][0]['label']}")
print(f"Labels match: {train[0]['label'] == repeat_split['train'][0]['label']}")
```

Output:

```
Train: 20000 | Test: 5000
Train[0] label: 1
Test[0] label: 0
Repeat train[0] label: 1
Labels match: True
```

**Versioning large files.** Datasets and model artifacts are too large for plain Git. Three options, in order of complexity:

- `.gitignore` the data directory entirely and document where to download it. Simplest, but not reproducible without manual steps.
- Git LFS tracks large files in Git. Works for files up to a few GB.
- DVC (Data Version Control) tracks datasets separately and stores metadata in Git. Handles large datasets and integrates with remote storage.

```bash
echo "data_log.jsonl" >> .gitignore
echo "valid_records.jsonl" >> .gitignore
echo "quarantine.jsonl" >> .gitignore
echo "*.parquet" >> .gitignore
echo "__pycache__/" >> .gitignore

git add .gitignore
git status
```

**Enforcing schemas at boundaries.** In production, the schema validation function runs at every input boundary: when data enters from an API, when a CSV is uploaded, when an enrichment provider returns results. The quarantine file is monitored — a spike in quarantine rate signals that an upstream source changed its format, which is schema drift. That is your signal to update the normalizer before the quarantine backlog grows.

For GTM pipelines specifically, this means validating before you send to Clay, validating before you write to CRM, and validating before you load into Smartlead. Each of those systems has its own input expectations. A schema violation at the Smartlead boundary — like a null email field — produces a bounced email that damages your sender reputation. [CITATION NEEDED — concept: Smartlead null email field produces bounce damaging sender reputation] The quarantine mechanism catches it before that happens.