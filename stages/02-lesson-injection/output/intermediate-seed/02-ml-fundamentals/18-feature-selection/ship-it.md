## Ship It

To ship a feature selection pipeline into a production lead scoring system, you need three artifacts: the selected feature list, the importance weights, and a scoring schema that maps features to enrichment sources.

The code below takes the output of our selection process and produces a deployable scoring configuration — the kind of JSON object that would sit between your enrichment layer and your CRM's score field. This is where Zone 2's data structure work meets the ML pipeline: the lead score is a JSON object whose keys are the features that survived selection.

```python
import json

selected_features = sorted(set(corr_selected) & set(mi_selected))
if len(selected_features) < 3:
    selected_features = sorted(set(corr_selected) | set(mi_selected))[:8]

feature_weights = {}
for name in selected_features:
    idx = int(name.split("_")[1])
    weight = float(max(correlations[name], mi_scores[name]))
    feature_weights[name] = round(weight, 4)

enrichment_sources = {}
source_pool = [
    "linkedin_employee_count",
    "crunchbase_funding_total",
    "builtwith_tech_stack",
    "g2_intent_score",
    "bombora_intent_topic",
    "apollo_revenue_estimate",
    "clearbit_industry",
    "webhook_engagement_score",
]
for i, name in enumerate(selected_features):
    enrichment_sources[name] = source_pool[i % len(source_pool)]

scoring_config = {
    "model_version": "1.0.0",
    "selection_method": "correlation + mutual_info intersection",
    "selected_features": selected_features,
    "feature_count": len(selected_features),
    "feature_weights": feature_weights,
    "enrichment_mapping": enrichment_sources,
    "score_formula": "weighted_sum / sum(weights) * 100",
    "thresholds": {
        "qualified": 65,
        "marketing_qualified": 45,
        "disqualified": 20,
    },
    "last_trained": "2024-01-15",
    "training_samples": 3000,
}

print("=== DEPLOYABLE SCORING CONFIG ===")
print(json.dumps(scoring_config, indent=2))
print()

print("=== PRODUCTION READINESS CHECK ===")
checks = {
    "Feature count < 15": len(selected_features) < 15,
    "Every feature has enrichment source": all(
        f in enrichment_sources for f in selected_features
    ),
    "Every feature has weight > 0": all(
        w > 0 for w in feature_weights.values()
    ),
    "Has qualified threshold": "qualified" in scoring_config["thresholds"],
    "Has model version": bool(scoring_config["model_version"]),
}

for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")

all_pass = all(checks.values())
print(f"\nReady to deploy: {all_pass}")
```

The scoring config is what you hand to your CRM integration or your Clay workflow. Each new account enters the pipeline, the enrichment layer populates the fields listed in `enrichment_mapping`, the scoring function computes a weighted sum, and the result lands as a single number in your CRM's lead score field. When you retrain the model on new win/loss data, you regenerate this config and redeploy — the schema is the contract between your ML pipeline and your GTM stack. [CITATION NEEDED — concept: Clay scoring formula field configuration]

One production caveat: feature selection can drift. A feature that was noise six months ago (say, "uses a specific CRM tool") might become predictive after a market shift. Rerun selection quarterly against fresh win/loss data and compare the selected feature set to the previous version. If more than 30% of features changed, your ICP definition may have shifted, and the scoring thresholds need recalibration.