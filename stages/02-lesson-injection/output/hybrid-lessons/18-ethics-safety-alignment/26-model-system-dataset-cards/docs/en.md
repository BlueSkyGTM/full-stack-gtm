# Model, System, and Dataset Cards

## Learning Objectives

- Write a valid model card as structured YAML and validate it against a schema with missing-field detection.
- Compare model, dataset, and system cards by scope, audience, and the questions each forces you to answer.
- Generate a dataset card with provenance, license constraints, and known biases for GTM enrichment data.
- Configure a CI check that fails when a card's `last_reviewed` date exceeds a configurable staleness threshold.
- Trace failure propagation through a multi-model pipeline using a system card's documented dependency graph.

## The Problem

A model ships to production. Three months later, nobody remembers what training data it used, what "fairness" meant to the team that built it, or why the decision threshold was set at 0.73 instead of 0.80. The original data scientist left. The Jupyter notebook that trained it lives in someone's local `~/projects/old_stuff/` directory. The README says "TODO: add docs." The model keeps running, making decisions about real prospects and real pipeline priorities, and nobody can answer a basic question: what is this thing actually doing, and is it still doing it correctly?

This is the institutional memory problem. It is not solved by writing more documentation — it is solved by writing *structured* documentation that answers fixed questions and lives alongside the artifact it describes. A README is free-form prose that decays. A card is a schema-constrained artifact that can be linted, diffed, version-controlled, and programmatically consumed.

The stakes are not just operational. Regulatory frameworks increasingly require traceability: what data trained this model, what metrics were evaluated, what groups were disaggregated in the analysis. [CITATION NEEDED — concept: EU AI Act documentation requirements for high-risk AI systems] Without structured cards, answering these questions after deployment means archaeology — digging through commit history, Slack threads, and tribal knowledge. Cards force the answers to exist at creation time, when the context is fresh.

## The Concept

Three card types exist because a deployed AI system has three distinct layers that each need their own documentation. Model cards document a single trained artifact: the weights, the benchmarks, the intended use, the failure modes. Dataset cards document the provenance, composition, and licensing of the data that trained or evaluated a model. System cards document the assembled pipeline: which models call which, what guardrails wrap them, how failure propagates from one component to the next.

The mechanism behind all three is constraint. A blank page invites omission; a schema with required fields prevents it. By forcing answers to fixed questions — what is the intended use, what is out of scope, what metrics were evaluated, what ethical review occurred — cards prevent the documentation drift that makes deployed systems unaccountable. Mitchell et al. (2019) introduced model cards with this exact framing: disaggregated evaluation across subgroups, intended use statements, and ethical considerations as required fields. Gebru et al. (2021) extended the pattern to datasets with datasheets, borrowing the analogy from electronics: every physical component ships with a datasheet specifying its characteristics, so why shouldn't every dataset?

```mermaid
flowchart LR
    subgraph DS["Dataset Card"]
        direction TB
        D1[Provenance & Sources]
        D2[Composition & Distribution]
        D3[License Constraints]
    end
    subgraph MD["Model Card"]
        direction TB
        M1[Weights & Architecture]
        M2[Disaggregated Evaluation]
        M3[Failure Modes]
    end
    subgraph SY["System Card"]
        direction TB
        S1[Pipeline Topology]
        S2[Guardrails & Limits]
        S3[Failure Propagation]
    end
    DS -->|trains| MD
    MD -->|assembled into| SY
    SY -->|deployed to| P[Production]
```

The scope distinction matters because each card type is consumed by a different audience. A model card is read by an ML engineer deciding whether to deploy or fine-tune a specific artifact. A dataset card is read by a data engineer or compliance officer evaluating whether the training data is licensed for their use case and whether its biases are acceptable for the target application. A system card is read by an SRE or security engineer who needs to understand how the whole pipeline behaves under failure — what happens when the enrichment API times out, when the LLM returns a malformed response, when the downstream scoring model receives partial input. System cards, as described by Sidhpurwala (2024) and elaborated in the "Blueprints of Trust" framework, cover end-to-end concerns: security capabilities, prompt-injection attack surfaces, data-exfiltration pathways, and alignment properties that no single model card can capture.

The empirical case for cards is measurable. Liang et al. (2024) found that model card detail on Hugging Face correlates with up to a 29% increase in downloads — not because documentation is marketing, but because practitioners can evaluate fitness for purpose without running the model first. Conversely, Oreamuno et al. (2023) found that only 0.3% of Hugging Face model cards document ethical considerations. The gap between what cards should contain and what they typically do contain is the gap this lesson addresses.

## Build It

Cards are structured data, not prose. That distinction is what makes them lintable, diffable, and consumable by automation. To demonstrate this, we will define a model card schema as a Python dictionary, write a model card as YAML, validate it against the schema, and print the parsed fields. No external validation library is needed — the constraint mechanism is simple enough to implement directly, which also makes it transparent.

```python
import yaml
from datetime import datetime, date

MODEL_CARD_SCHEMA = {
    "required": [
        "name", "version", "owner", "last_reviewed",
        "model_type", "intended_use", "out_of_scope_use",
        "training_data", "evaluation", "ethical_considerations",
        "caveats"
    ],
    "evaluation_required": ["metric", "value", "threshold"],
    "training_data_required": ["source", "n_samples", "label_definition"],
}

model_card_yaml = """
name: lead-score-v3
version: 3.1.0
owner: gtm-engineering
last_reviewed: '2025-01-15'
model_type: logistic_regression
intended_use: >
  Score inbound leads on likelihood to convert to SQL
  within 30 days, used to prioritize SDR follow-up queue.
out_of_scope_use:
  - Scoring outbound prospect lists
  - Cross-sell recommendations for existing customers
training_data:
  source: salesforce_crm_export_2024Q3
  n_samples: 48200
  label_definition: converted_to_sql_within_30d
evaluation:
  metric: auc_roc
  value: 0.81
  threshold: 0.73
  threshold_justification: >
    Set at 90th percentile of training conversion scores.
    Below this, precision drops below 0.35.
ethical_considerations:
  - Model trained on historical CRM data that under-represents
    companies founded after 2022.
known_biases:
  - Industry: overweights SaaS companies due to training data skew.
caveats:
  - Retrain quarterly. Score drift detected after 60+ days.
"""

card = yaml.safe_load(model_card_yaml)

errors = []

for field in MODEL_CARD_SCHEMA["required"]:
    if field not in card:
        errors.append(f"Missing top-level field: {field}")

if "evaluation" in card:
    for field in MODEL_CARD_SCHEMA["evaluation_required"]:
        if field not in card["evaluation"]:
            errors.append(f"Missing evaluation subfield: {field}")

if "training_data" in card:
    for field in MODEL_CARD_SCHEMA["training_data_required"]:
        if field not in card["training_data"]:
            errors.append(f"Missing training_data subfield: {field}")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
    raise SystemExit(1)

review_date = datetime.strptime(str(card["last_reviewed"]), "%Y-%m-%d").date()
age_days = (date.today() - review_date).days

print("=" * 55)
print(f"MODEL CARD: {card['name']} v{card['version']}")
print("=" * 55)
print(f"Owner:          {card['owner']}")
print(f"Type:           {card['model_type']}")
print(f"Last reviewed:  {card['last_reviewed']} ({age_days} days ago)")
print(f"Intended use:   {card['intended_use'].strip()}")
print(f"Eval metric:    {card['evaluation']['metric']} = {card['evaluation']['value']}")
print(f"Threshold:      {card['evaluation']['threshold']}")
print(f"Out-of-scope:   {len(card['out_of_scope_use'])} use cases")
print(f"Training data:  {card['training_data']['n_samples']:,} samples from {card['training_data']['source']}")
print(f"Biases doc'd:   {len(card.get('known_biases', []))} items")
print("=" * 55)
print("STATUS: VALID")
```

Run this and you get a formatted report showing every parsed field. Now break it: remove the `threshold` line from the evaluation block and run again. The validator catches the missing subfield and exits with a non-zero code. That is the mechanism. The card is not valid prose — it is a structured artifact that either satisfies the schema or does not. This is what makes it consumable by CI pipelines, not just by humans.

The schema above is deliberately hand-rolled for transparency. In production, you would use a formal schema format (JSON Schema) and a validation library. But the principle is identical: define required fields, validate against them, fail loudly when something is missing.

## Use It

Dataset cards exist because the data feeding a model determines its behavior more than the model architecture does. In GTM engineering, this is acutely visible: the Clay waterfall enrichment pipeline pulls firmographic data from multiple upstream sources — Clearbit, ZoomInfo, LinkedIn — and each source has its own coverage gaps, licensing constraints, and biases. When you swap one data source for another in the waterfall, your ICP scores shift. Without a dataset card documenting what the training data actually contained, you cannot explain why.

This maps directly to Zone 18 of the GTM stack: advanced prompting and CoT for multi-step research chains, where an agent reasons about an account before writing outreach. The dataset card is the documentation layer behind that reasoning. If your agent's account research draws on enrichment data whose provenance and biases are undocumented, the chain-of-thought output inherits those biases invisibly. The dataset card makes the data's limitations explicit so the agent's reasoning can account for them — or so a human reviewer can flag when it does not.

```python
import yaml
from datetime import datetime, date

dataset_card_yaml = """
name: prospect-firmographic-enrichment-2025q1
version: 1.0.0
owner: gtm-engineering
last_reviewed: '2025-01-20'
dataset_type: firmographic_enrichment
provenance:
  primary_source: clay_waterfall_enrichment
  upstream_sources:
    - name: clearbit
      fields: [employee_count, industry, alexa_rank]
      coverage_pct: 78
    - name: zoominfo
      fields: [revenue_range, tech_stack]
      coverage_pct: 64
    - name: linkedin_company_api
      fields: [headquarters, founded_date, employee_growth]
      coverage_pct: 71
  collection_method: api_enrichment_pipeline
  collection_date_range: '2025-01-01 to 2025-01-15'
  n_records: 125000
composition:
  geo_distribution:
    north_america: 62
    emea: 24