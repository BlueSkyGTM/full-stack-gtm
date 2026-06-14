## Ship It

### Easy: Format on Save Verification

Configure your editor to format Python on save (the settings JSON from the Build It section does this). Then create this script, format it manually with Black to confirm the configuration is working, and run it:

```python
x = {"b": 2, "a": 1}
y = sorted(x.items())
print(f"formatted: {dict(y)}")
```

Run:

```bash
black format_check.py && python format_check.py
```

Expected output:

```
reformatted format_check.py
formatted: {'a': 1, 'b': 2}
```

If Black reports "1 file reformatted," your formatter is working. If it says "1 file left unchanged," the file was already formatted—modify it to be poorly formatted (bad indentation, extra spaces) and run again to confirm Black catches it.

### Medium: Enrichment Payload Schema Validation

Write a JSON schema for a webhook payload that an enrichment service sends to your endpoint after processing a company. The payload must include a `company` object with `domain` (string) and `name` (string), a `founded_year` (integer, minimum 1800), and an `enrichment_status` (enum: "complete", "partial", "failed"). Validate two payloads—one valid, one with `enrichment_status` set to "success" (not in the enum)—and print the results:

```python
import json
from jsonschema import validate, ValidationError

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["company", "founded_year", "enrichment_status"],
    "properties": {
        "company": {
            "type": "object",
            "required": ["domain", "name"],
            "properties": {
                "domain": {"type": "string"},
                "name": {"type": "string"}
            }
        },
        "founded_year": {"type": "integer", "minimum": 1800},
        "enrichment_status": {
            "type": "string",
            "enum": ["complete", "partial", "failed"]
        }
    }
}

valid = {
    "company": {"domain": "clay.com", "name": "Clay"},
    "founded_year": 2017,
    "enrichment_status": "complete"
}

invalid = {
    "company": {"domain": "clay.com", "name": "Clay"},
    "founded_year": 2017,
    "enrichment_status": "success"
}

for label, payload in [("valid", valid), ("invalid", invalid)]:
    try:
        validate(instance=payload, schema=schema)
        print(f"{label}: PASS")
    except ValidationError as e:
        print(f"{label}: FAIL — {e.message}")
```

Output:

```
valid: PASS
invalid: FAIL — 'success' is not one of ['complete', 'partial', 'failed']
```

### Hard: Multi-File Project with Combined Validation

Create a project with two files: a Python module and a JSON config. Configure a single command that runs the linter, formatter check, and schema validation, then prints a combined summary.

Project structure:

```
project/
├── config.json
├── config_schema.json
├── pipeline.py
└── validate_all.py
```

`config_schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["enrichment_provider", "scoring_endpoint"],
  "properties": {
    "enrichment_provider": {"type": "string", "enum": ["clay", "clearbit", "zoominfo"]},
    "scoring_endpoint": {"type": "string"},
    "max_companies_per_batch": {"type": "integer", "minimum": 1, "maximum": 1000}
  }
}
```

`config.json`:

```json
{
  "enrichment_provider": "clay",
  "scoring_endpoint": "https://api.scoring.example.com/v1/score",
  "max_companies_per_batch": 100
}
```

`pipeline.py`:

```python
def enrich_and_score(domain: str, provider: str) -> dict:
    return {"domain": domain, "provider": provider, "score": 0.85}


if __name__ == "__main__":
    result = enrich_and_score("clay.com", "clay")
    print(result)
```

`validate_all.py`:

```python
import json
import subprocess
import sys
from jsonschema import validate, ValidationError

results = []

with open("config_schema.json") as f:
    schema = json.load(f)
with open("config.json") as f:
    config = json.load(f)

try:
    validate(instance=config, schema=schema)
    results.append(("config.json schema validation", "PASS"))
except ValidationError as e:
    results.append(("config.json schema validation", f"FAIL: {e.message}"))

fmt_result = subprocess.run(
    ["black", "--check", "pipeline.py"],
    capture_output=True, text=True
)
if fmt_result.returncode == 0:
    results.append(("pipeline.py formatting", "PASS"))
else:
    results.append(("pipeline.py formatting", "FAIL: needs formatting"))

lint_result = subprocess.run(
    ["ruff", "check", "pipeline.py"],
    capture_output=True, text=True
)
if lint_result.returncode == 0:
    results.append(("pipeline.py linting", "PASS"))
else:
    results.append(("pipeline.py linting", f"FAIL: {lint_result.stdout.strip()}"))

print("=" * 50)
print("VALIDATION SUMMARY")
print("=" * 50)
all_pass = True
for check, status in results:
    marker = "✓" if status == "PASS" else "✗"
    print(f"  {marker} {check}: {status}")
    if status != "PASS":
        all_pass = False
print("=" * 50)
print(f"Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
sys.exit(0 if all_pass else 1)
```

Run:

```bash
black pipeline.py && python validate_all.py
```

Output:

```
==================================================
VALIDATION SUMMARY
==================================================
  ✓ config.json schema validation: PASS
  ✓ pipeline.py formatting: PASS
  ✓ pipeline.py linting: PASS
==================================================
Overall: ALL PASS
```

This is the workflow you run before committing changes to a GTM integration. Schema validation catches malformed config. Formatting ensures clean diffs. Linting catches unused imports and undefined names. One command, three checks, zero excuses for shipping a broken payload.