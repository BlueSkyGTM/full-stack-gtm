## Ship It

Spec versioning is the difference between a prompt you can trust in production and a prompt you are afraid to touch. When you modify a constraint, add a field to the output schema, or rewrite the context block, you bump `spec_version`. The version string lets you answer the question every operator eventually faces: "the enrichment results changed on Thursday — did we change the prompt, or did the model behave differently?" Without version tracking, you cannot answer that question. With it, you diff the spec and know immediately.

Here is spec version 2, which tightens the employee count range and adds a revenue field to the output:

```python
TASK_SPEC_V2 = {
    "spec_version": "2.0.0",
    "task_id": "tam_refinement_classifier",
    "role": "You are a B2B SaaS market analyst evaluating companies for outbound prospecting.",
    "objective": "Classify whether the given company is a valid prospect for a sales engagement platform seller.",
    "context": (
        "The seller provides sales engagement software priced for mid-market companies. "
        "Valid prospects have 100-500 employees (tightened from 50 in v1), use Salesforce or HubSpot, "
        "and operate in SaaS, financial services, insurance, logistics, or consulting. "
        "Companies below 100 employees are too small to justify the seat count."
    ),
    "constraints": [
        "Respond ONLY with a JSON object matching the output schema.",
        "Do not include markdown formatting, code fences, or explanations.",
        "If you cannot determine a field value, set it to null.",
        "The reason field must be 50 words or fewer.",
        "Estimate annual_revenue_range based on employee_count if revenue data is absent."
    ],
    "input_schema": {
        "company_name": "string",
        "employee_count": "integer or null",
        "crm_used": "string or null",
        "industry": "string or null",
        "website": "string or null"
    },
    "output_schema": {
        "is_prospect": "boolean",
        "confidence": "float between 0.0 and 1.0",
        "reason": "string, max 50 words",
        "disqualifying_factor": "string or null",
        "estimated_revenue_range": "string, one of: '<$5M', '$5M-$25M', '$25M-$100M', '>$100M'"
    },
    "examples": [
        {
            "input": {
                "company_name": "Acme Logistics",
                "employee_count": 120,
                "crm_used": "Salesforce",
                "industry": "Logistics",
                "website": "acmelogistics.com"
            },
            "output": {
                "is_prospect": True,
                "confidence": 0.9,
                "reason": "Mid-market size, uses Salesforce, logistics is addressable.",
                "disqualifying_factor": None,
                "estimated_revenue_range": "$5M-$25M"
            }
        },
        {
            "input": {
                "company_name": "Bob's Bakery",
                "employee_count": 8,
                "crm_used": None,
                "industry": "Food Service",
                "website": "bobsbakery.com"
            },
            "output": {
                "is_prospect": False,
                "confidence": 0.95,
                "reason": "Below 100-employee minimum. No CRM detected. Food service is not addressable.",
                "disqualifying_factor": "employee_count below threshold",
                "estimated_revenue_range": "<$5M"
            }
        }
    ]
}
```

The regression test runs the same inputs through both versions and reports which predictions changed:

```python
def run_regression(spec_v1, spec_v2, test_inputs):
    print(f"REGRESSION TEST: {spec_v1['spec_version']} vs {spec_v2['spec_version']}")
    print(f"Inputs: {len(test_inputs)}")
    print("=" * 70)

    results = []
    for inp in test_inputs:
        r1 = mock_classify_silent(spec_v1, inp)
        r2 = mock_classify_silent(spec_v2, inp)
        changed = r1["is_prospect"] != r2["is_prospect"]
        results.append({
            "company": inp["company_name"],
            "v1_prospect": r1["is_prospect"],
            "v2_prospect": r2["is_prospect"],
            "changed": changed,
            "v1_conf": r1["confidence"],
            "v2_conf": r2["confidence"]
        })

    print(f"{'Company':<25} {'v1':<8} {'v2':<8} {'Changed':<10} {'v1 conf':<10} {'v2 conf'}")
    print("-" * 70)
    for r in results:
        print(f"{r['company']:<25} {str(r['v1_prospect']):<8} {str(r['v2_prospect']):<8} {str(r['changed']):<10} {r['v1_conf']:<10} {r['v2_conf']}")

    changed_count = sum(1 for r in results if r["changed"])
    print("-" * 70)
    print(f"CHANGED PREDICTIONS: {changed_count}/{len(results)}")
    print(f"VERSION DIFF: {spec_v1['spec_version']} -> {spec_v2['spec_version']}")
    print(f"KEY CHANGE: employee_count threshold moved from 50 to 100")
    return results

def mock_classify_silent(spec, input_data):
    ec = input_data.get("employee_count")
    crm = input_data.get("crm_used")
    industry = input_data.get("industry")

    if spec["spec_version"].startswith("1"):
        min_emp = 50
    else:
        min_emp = 100

    valid_size = ec is not None and min_emp <= ec <= 500
    valid_crm = crm is not None and crm.lower() in ("salesforce", "hubspot")
    valid_industry = industry is not None and industry.lower() in (
        "saas", "software", "financial services", "insurance",
        "logistics", "consulting", "technology"
    )

    is_prospect = valid_size and valid_crm and valid_industry
    return {
        "is_prospect": is_prospect,
        "confidence": round(0.3 + 0.2 * sum([valid_size, valid_crm, valid_industry]), 2)
    }

test_inputs = [
    {"company_name": "Globex Corp", "employee_count": 340, "crm_used": "HubSpot", "industry": "SaaS"},
    {"company_name": "SmallShop Inc", "employee_count": 75, "crm_used": "Salesforce", "industry": "Consulting"},
    {"company_name": "TinyStartup", "employee_count": 12, "crm_used": None, "industry": "Retail"},
    {"company_name": "MidTier Co", "employee_count": 250, "crm_used": "Salesforce", "industry": "Insurance"},
    {"company_name": "Edge Case LLC", "employee_count": 80, "crm_used": "HubSpot", "industry": "Logistics"},
]

run_regression(TASK_SPEC_V1, TASK_SPEC_V2, test_inputs)
```

SmallShop Inc and Edge Case LLC flip from prospect to non-prospect between versions because the employee threshold moved from 50 to 100. That is the regression signal — 2 out of 5 inputs changed classification. If that delta is acceptable, you ship v2. If those companies were your best customers, you roll back or adjust the threshold.

Over-constraining is the other failure mode to watch for. A spec with twenty constraints, a rigid output schema, and zero tolerance for null fields sounds disciplined but produces two problems. First, the model spends attention budget on constraint compliance instead of reasoning about the actual classification, which degrades accuracy on edge cases. Second, real-world inputs are messy — Apollo exports have null employee counts, wrong industries, and missing CRM data. A spec that rejects every input with a null field will classify nothing. The constraint block should specify what to do with missing data (use null, estimate, or flag) rather than forbidding it.