## Ship It

To ship this into a production enrichment pipeline, you need batch processing, a scoring threshold that maps to a routing decision, and handling for the edge case where the model returns low confidence across all criteria. The following script processes a list of leads, scores each against your ICP hypotheses, and emits a structured qualification verdict that you could write to a CRM field or use to trigger a workflow.

```python
from transformers import pipeline
import json

classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

icp_hypotheses = [
    "This team needs data unification",
    "This team has manual workflow inefficiency",
    "This team needs custom integration work",
    "This team is actively evaluating tools",
]

ENTAIL_THRESHOLD = 0.75
POSSIBLE_THRESHOLD = 0.40

leads = [
    {"name": "Acme Corp", "email": "ops@acme.com",
     "intake": "We have customer data scattered across HubSpot, three Google Sheets, and an Airtable base."},
    {"name": "Globex", "email": "it@globex.com",
     "intake": "Our current data warehouse works fine. Just exploring options."},
    {"name": "Initech", "email": "eng@initech.com",
     "intake": "We need custom API integrations between Salesforce and our billing system."},
    {"name": "Umbrella", "email": "data@umbrella.com",
     "intake": "Looking for a platform that can pull from 12 data sources and auto-generate reports."},
]

results = []
for lead in leads:
    scores = classifier(lead["intake"], icp_hypotheses, multi_label=True)
    score_dict = dict(zip(scores["labels"], scores["scores"]))
    strong = [l for l, s in score_dict.items() if s > ENTAIL_THRESHOLD]
    possible = [l for l, s in score_dict.items() if POSSIBLE_THRESHOLD < s <= ENTAIL_THRESHOLD]

    if len(strong) >= 2:
        verdict = "PRIORITY"
    elif len(strong) >= 1:
        verdict = "QUALIFIED"
    elif possible:
        verdict = "NURTURE"
    else:
        verdict = "DISQUALIFY"

    results.append({
        "name": lead["name"],
        "email": lead["email"],
        "verdict": verdict,
        "strong_signals": strong,
        "possible_signals": possible,
        "top_score": max(score_dict.values()),
    })

print(json.dumps(results, indent=2))
```

Output:

```json
[
  {
    "name": "Acme Corp",
    "email": "ops@acme.com",
    "verdict": "PRIORITY",
    "strong_signals": [
      "This team needs data unification",
      "This team has manual workflow inefficiency"
    ],
    "possible_signals": [],
    "top_score": 0.972
  },
  {
    "name": "Globex",
    "email": "it@globex.com",
    "verdict": "DISQUALIFY",
    "strong_signals": [],
    "possible_signals": [],
    "top_score": 0.388
  },
  {
    "name": "Initech",
    "email": "eng@initech.com",
    "verdict": "QUALIFIED",
    "strong_signals": [
      "This team needs custom integration work"
    ],
    "possible_signals": [],
    "top_score": 0.901
  },
  {
    "name": "Umbrella",
    "email": "data@umbrella.com",
    "verdict": "PRIORITY",
    "strong_signals": [
      "This team needs data unification",
      "This team is actively evaluating tools"
    ],
    "possible_signals": [],
    "top_score": 0.94
  }
]
```

The thresholds (0.75 for strong, 0.40 for possible) are starting points, not golden numbers. Calibrate them against a held-out set of leads where you know the ground-truth outcome — did the lead ultimately buy? Lower the entailment threshold if you are losing qualified leads to false neutrals; raise it if your sales team is drowning in low-quality passes. The verdict logic encodes your GTM routing rules: PRIORITY leads with two or more strong signals go straight to an AE, QUALIFIED leads with one signal go to SDR for discovery, NURTURE leads enter a sequence, DISQUALIFY leads are suppressed from outbound. A CRM with 10,000 contacts that has not been maintained will accumulate dead leads, bounced emails, and stale firmographic data — running this pipeline quarterly on your existing database surfaces buried opportunities that keyword filters would miss.

The same pipeline applies to post-outreach analysis. When a rep sends a follow-up email, you can check whether the email's content entails the specific buyer needs the prospect raised in discovery. Premise: the rep's email. Hypothesis: "This message addresses the buyer's concern about SOC 2 compliance." If the model returns neutral instead of entailment, the rep moved on without addressing the stated need — and you catch it before the prospect ghosts. [CITATION NEEDED — concept: Clay enrichment column using NLI-style entailment for ICP matching]