## Ship It

In production, three things change from the examples above. First, the validator gets more specific — instead of just checking field presence and type, it checks value constraints: `employee_range` must be one of `["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]`, `industry` must be one of your accepted industry taxonomy values, `company_name` must not be empty or all whitespace. Second, the feedback loop logs every attempt to persistent storage — not just to stdout — so you can analyze failure patterns across thousands of records and identify whether specific input types systematically require more retries. Third, the termination condition becomes cost-aware: the loop checks remaining budget (API credits, token budget, wall-clock time) and terminates early if the budget is exhausted, even if attempts remain.

The production pattern also separates the feedback loop from the business logic. The loop returns a structured result — either `{"success": True, "data": ..., "attempts": n}` or `{"success": False, "errors": [...], "attempts": n}` — and the calling code decides what to do with failures. For enrichment, a failed extraction might write `null` to the CRM and flag the record for manual review. For a classification pipeline, a failure might fall back to a default label. The loop does not make this decision. The loop detects failure, tries to correct it, and reports the outcome. What happens next is a business rule.

Here is a production-shaped validator with value constraints and structured logging:

```python
import json
from datetime import datetime, timezone

VALID_RANGES = {"1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"}
VALID_INDUSTRIES = {
    "SaaS", "Fintech", "Healthcare", "E-Commerce", "Logistics",
    "Manufacturing", "Media", "Education", "Real Estate", "Other",
}

def validate_company(data):
    errors = []
    for field in ("company_name", "industry", "employee_range"):
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(data[field], str) or not data[field].strip():
            errors.append(f"Field {field} must be a non-empty string")
    if "employee_range" in data and data["employee_range"] not in VALID_RANGES:
        errors.append(
            f"employee_range must be one of {sorted(VALID_RANGES)}, "
            f"got: {data['employee_range']}"
        )
    if "industry" in data and data["industry"] not in VALID_INDUSTRIES:
        errors.append(
            f"industry must be one of {sorted(VALID_INDUSTRIES)}, "
            f"got: {data['industry']}"
        )
    return errors

def enrichment_loop(
    extract_fn, raw_text, entity_id, max_attempts=3
):
    base_prompt = (
        "Extract company data as JSON with fields: "
        "company_name, industry (one of: "
        + ", ".join(sorted(VALID_INDUSTRIES))
        + "), employee_range (one of: "
        + ", ".join(sorted(VALID_RANGES))
        + ").\n\nText: " + raw_text
    )
    log = []
    error_history = []

    for attempt in range(max_attempts):
        prompt = base_prompt
        if error_history:
            prompt += (
                "\n\nPrevious errors:\n"
                + "\n".join(f"- {e}" for e in error_history[-1])
            )
        raw_output = extract_fn(prompt)
        try:
            data = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError) as e:
            data = {}
            error_history.append([f"JSON parse error: {e}"])
            log.append({"entity_id": entity_id, "attempt": attempt + 1,
                        "status": "parse_error", "raw": str(raw_output)[:200]})
            continue

        errors = validate_company(data)
        log.append({
            "entity_id": entity_id,
            "attempt": attempt + 1,
            "status": "pass" if not errors else "validation_error",
            "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if not errors:
            return {"success": True, "data": data, "log": log}

        error_history.append(errors)

    return {"success": False, "errors": error_history[-1], "log": log}

def mock_extract(prompt):
    if "Previous errors" in prompt:
        return json.dumps({
            "company_name": "TestCo",
            "industry": "SaaS",
            "employee_range": "11-50",
        })
    return json.dumps({
        "company_name": "TestCo",
        "industry": "Software",
        "employee_range": "50 people",
    })

result = enrichment_loop(
    mock_extract,
    "TestCo is a software company with about 50 employees.",
    entity_id="rec_001",
)

print(json.dumps(result, indent=2))
```

Attempt 1 fails because `"Software"` is not in the valid industry set and `"50 people"` is not a valid range. Attempt 2 succeeds because the feedback context told the model the exact valid values. The log records both attempts with timestamps, which you would persist to analyze extraction quality across your full dataset over time.