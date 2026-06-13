# Capstone — Build a Complete Tool Ecosystem

## Hook

You've used individual tools in isolation. Now you'll wire them together into a system where data flows from prospect identification through enrichment to outreach — and handle the failures that real APIs introduce.

## Concept

Integrating multiple APIs creates failure modes that don't exist in single-tool scripts: rate limits stack, data shapes drift, and one broken step corrupts everything downstream. The capstone pattern is a pipeline with fallbacks at each stage: primary source fails → secondary source → graceful degradation with logging. Every integration point needs a contract (expected schema) and a circuit breaker (stop calling after N failures).

## Demo

Build a three-stage pipeline: (1) fetch company data from a public API, (2) enrich each record with a second data source, (3) format results into a structured output. Each stage runs independently, logs its own errors, and writes intermediate state to disk so a failure at stage 3 doesn't require re-running stage 1.

```python
import json
import time
import os
from datetime import datetime

def fetch_companies():
    companies = [
        {"domain": "example.com", "name": "Example Corp", "employees": 250},
        {"domain": "test.org", "name": "Test Foundation", "employees": 80},
        {"domain": "demo.io", "name": "Demo Labs", "employees": 1200},
        {"domain": "sample.net", "name": "Sample Systems", "employees": 45},
        {"domain": "acme.dev", "name": "Acme Builders", "employees": 310},
    ]
    timestamp = datetime.now().isoformat()
    print(f"[STAGE 1] Fetched {len(companies)} companies at {timestamp}")
    with open("stage1_companies.json", "w") as f:
        json.dump({"data": companies, "fetched_at": timestamp}, f, indent=2)
    return companies

def enrich_company(company, index):
    time.sleep(0.1)
    enrichment = {
        "tech_stack": ["Python", "AWS"] if company["employees"] > 100 else ["Node.js", "GCP"],
        "funding_stage": "Series B" if company["employees"] > 500 else "Series A",
        "intent_score": round(0.3 + (index * 0.13), 2),
    }
    enriched = {**company, **enrichment}
    enriched["enrichment_source"] = "mock_api_v1"
    enriched["enriched_at"] = datetime.now().isoformat()
    return enriched

def run_enrichment(companies):
    enriched = []
    failures = []
    for i, company in enumerate(companies):
        try:
            result = enrich_company(company, i)
            enriched.append(result)
        except Exception as e:
            failures.append({"domain": company["domain"], "error": str(e)})
            enriched.append({**company, "enrichment_error": str(e)})
    print(f"[STAGE 2] Enriched {len(enriched) - len(failures)}/{len(companies)} records, {len(failures)} failures")
    with open("stage2_enriched.json", "w") as f:
        json.dump({"data": enriched, "failures": failures}, f, indent=2)
    return enriched, failures

def generate_outreach(enriched_records):
    outputs = []
    for record in enriched_records:
        if "enrichment_error" in record:
            continue
        tier = "enterprise" if record["employees"] > 500 else "mid-market" if record["employees"] > 100 else "smb"
        message = f"Hi {{first_name}}, noticed {record['name']} uses {', '.join(record['tech_stack'])}. Our platform complements that stack for {tier}-scale teams."
        outputs.append({
            "domain": record["domain"],
            "company": record["name"],
            "tier": tier,
            "intent_score": record["intent_score"],
            "message_template": message,
        })
    print(f"[STAGE 3] Generated {len(outputs)} outreach records (skipped {len(enriched_records) - len(outputs)} incomplete)")
    with open("stage3_outreach.json", "w") as f:
        json.dump({"data": outputs, "generated_at": datetime.now().isoformat()}, f, indent=2)
    return outputs

def run_pipeline():
    print("=" * 60)
    print("PIPELINE START")
    print("=" * 60)
    companies = fetch_companies()
    enriched, failures = run_enrichment(companies)
    outreach = generate_outreach(enriched)
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Input:    {len(companies)} companies")
    print(f"  Enriched: {len(enriched) - len(failures)} successful")
    print(f"  Outreach: {len(outreach)} ready")
    print(f"  Drop-off: {len(companies) - len(outreach)} records lost in pipeline")
    print("=" * 60)
    return outreach

results = run_pipeline()
print("\nSample output:")
print(json.dumps(results[0], indent=2))
```

## Use It

This is the Clay waterfall pattern generalized: fetch → enrich → act, with each stage persisted independently. In GTM, this maps directly to Zone 02 (Enrich) feeding Zone 03 (Engage), where enrichment failures determine whether a prospect gets a personalized outreach track or falls to a generic fallback sequence. The intermediate JSON files mirror what Clay's waterfall stores per-cell — the state that lets you resume without re-fetching.

**Exercise hooks:**
- *Easy*: Add a fourth stage that writes outreach records to a CSV file with columns: domain, company, tier, intent_score.
- *Medium*: Modify `enrich_company` to randomly fail 30% of the time; confirm the pipeline completes and logs which records failed.
- *Hard*: Add a resume function that detects existing `stage2_enriched.json` and skips stage 1–2 on re-run, only regenerating stage 3.

## Ship It

Replace mock data with real API calls: swap `fetch_companies` for a Crunchbase or Apollo API call, swap `enrich_company` for a Clearbit or Hunter.io lookup, and swap `generate_outreach` for a Slack webhook or email send via Resend. Deploy the pipeline as a scheduled cron job or GitHub Action that runs nightly and writes results to a shared Google Sheet or Notion database.

**Exercise hooks:**
- *Easy*: Wire one real API (any free-tier endpoint) into a single stage and confirm the pipeline still completes end-to-end.
- *Medium*: Add error retry logic: if a single company's enrichment fails, retry up to 3 times with exponential backoff before marking it as a failure.
- *Hard*: Build a CLI that accepts `--stage 1|2|3|all` and `--resume true|false`, runs only the requested stages, and respects the resume flag by checking for intermediate files.

## Evaluate

**Reflection questions** (not a quiz — capstone assessment is demonstration-based):
1. Where in your pipeline does a single API failure cause the most downstream damage, and how does your design mitigate that?
2. If you needed to process 10,000 companies instead of 5, what bottleneck appears first — rate limits, memory, or disk I/O?
3. Your enrichment API returns a different schema on Tuesday than it did on Monday. How does your pipeline detect and respond to that drift?

**Deliverable check**: Your pipeline must produce `stage1_companies.json`, `stage2_enriched.json`, and `stage3_outreach.json` with consistent schemas, and it must complete without error even when individual records fail enrichment.