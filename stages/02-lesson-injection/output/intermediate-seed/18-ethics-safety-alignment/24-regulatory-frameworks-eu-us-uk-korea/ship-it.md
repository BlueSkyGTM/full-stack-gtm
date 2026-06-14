## Ship It

Now build the compliance gate that sits in your GTM pipeline. Every prospect record passes through this gate before any enrichment or outreach fires. The gate reads the prospect's jurisdiction, checks the processing activity against the applicable framework, and either permits, blocks, or flags the record.

```python
import json
from datetime import datetime


JURISDICTION_RULES = {
    "EU": {
        "countries": ["DE", "FR", "NL", "IE", "ES", "IT", "PL", "SE", "AT", "BE"],
        "permitted_fields_without_consent": [
            "company_name", "company_domain", "industry", "employee_count",
        ],
        "requires_consent": ["personal_email", "phone", "linkedin_url"],
        "requires_dpia": ["behavioral_profiling", "ml_scoring_individual_level"],
        "requires_human_review": ["automated_decision_affecting_access"],
    },
    "UK": {
        "countries": ["GB"],
        "permitted_fields_without_consent": [
            "company_name", "company_domain", "industry", "employee_count",
        ],
        "requires_consent": ["personal_email", "phone"],
        "requires_dpia": ["behavioral_profiling"],
        "requires_human_review": ["automated_decision_affecting_access"],
    },
    "KR": {
        "countries": ["KR"],
        "permitted_fields_without_consent": ["company_name", "company_domain"],
        "requires_consent": [
            "personal_email", "phone", "linkedin_url",
            "industry", "employee_count", "job_title",
        ],
        "requires_dpia": ["ml_scoring_individual_level"],
        "requires_human_review": ["ml_scoring_individual_level"],
        "requires_overseas_transfer_consent": True,
    },
    "US": {
        "countries": ["US"],
        "permitted_fields_without_consent": [
            "company_name", "company_domain", "industry",
            "employee_count", "job_title", "personal_email",
            "phone", "linkedin_url",
        ],
        "requires_consent": [],
        "requires_dpia": [],
        "requires_human_review": [],
        "state_overrides": {
            "CA": {
                "requires_opt_out": ["data_sale", "data_sharing"],
                "requires_deletion_handler": True,
            },
            "CO": {
                "requires_assessment": ["high_risk_ai"],
            },
        },
    },
}

AUDIT_LOG = []


def classify_jurisdiction(country_code):
    cc = country_code.upper()
    for jurisdiction, rules in JURISDICTION_RULES.items():
        if cc in rules["countries"]:
            if jurisdiction == "US" and "state_overrides":
                return jurisdiction, cc
            return jurisdiction, cc
    return "UNKNOWN", cc


def get_state_code(country_code):
    return None


def validate_record(record, processing_activity):
    country = record.get("country_code", "US")
    jurisdiction, cc = classify_jurisdiction(country)

    state = record.get("state_code")
    fields_present = set(record.get("data_fields", {}).keys())
    fields_stored = record.get("data_fields", {})

    result = {
        "record_id": record.get("id", "unknown"),
        "jurisdiction": jurisdiction,
        "country": cc,
        "state": state,
        "decision": "PERMIT",
        "required_actions": [],
        "blocked_fields": [],
        "warnings": [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    rules = JURISDICTION_RULES.get(jurisdiction)
    if not rules:
        result["decision"] = "REVIEW"
        result["warnings"].append(f"Unknown jurisdiction: {cc}. Manual review required.")
        log_decision(result)
        return result

    consent_given = record.get("consent_given", False)

    for field in fields_present:
        if field in rules.get("requires_consent", []) and not consent_given:
            result["blocked_fields"].append(field)
            result["decision"] = "BLOCK"
            result["required_actions"].append(
                f"Obtain consent for field '{field}' before processing under {jurisdiction} rules"
            )

    if jurisdiction == "KR" and rules.get("requires_overseas_transfer_consent"):
        if not record.get("overseas_transfer_consent", False):
            result["decision"] = "BLOCK"
            result["required_actions"].append(
                "Obtain separate consent for overseas data transfer (PIPA)"
            )

    if processing_activity in rules.get("requires_dpia", []):
        if not record.get("dpia_completed", False):
            result["decision"] = "BLOCK" if result["decision"] == "BLOCK" else "FLAG"
            result["required_actions"].append(
                f"Conduct DPIA before {processing_activity} under {jurisdiction} rules"
            )

    if processing_activity in rules.get("requires_human_review", []):
        if not record.get("human