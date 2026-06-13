# Compliance — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

---

## Learning Objectives

1. Map data flows through GTM systems to identify which compliance frameworks apply based on data type, geography, and processing activity.
2. Implement audit logging and data classification that produces SOC 2 Type II evidence artifacts.
3. Configure retention and deletion pipelines that enforce GDPR right-to-erasure on prospect and customer records.
4. Classify AI-powered GTM features by risk tier under the EU AI Act and ISO 42001.
5. Build a compliance-as-code pipeline that generates auditor-ready evidence from infrastructure and application logs.

---

## Beat 1: Hook — Your GTM Stack Is a Compliance Surface

Your outbound engine collects email addresses, firmographic data, behavioral signals, and—increasingly—AI-generated content. Every one of those data categories triggers obligations under at least one framework. A single unchecked webhook leaking PII to an unvetted vendor can invalidate your SOC 2 report, trigger GDPR fines up to 4% of global revenue, and kill enterprise deals during security review. This lesson maps the six frameworks that matter for GTM engineering and builds the controls that satisfy all of them from shared primitives.

---

## Beat 2: Learn It — Control Families and Where They Overlap

Each framework is a set of controls—policies, technical measures, and evidence requirements—that an auditor or regulator verifies. The key insight: you do not implement six separate compliance programs. You implement a shared control base (access logging, encryption, data classification, retention enforcement, vendor management, incident response) and then add framework-specific extensions.

**Core control mapping:**

| Control | SOC 2 | HIPAA | GDPR | PCI-DSS | EU AI Act | ISO 42001 |
|---|---|---|---|---|---|---|
| Access logging | ✓ CC6.1 | ✓ §164.312(b) | ✓ Art. 30 | ✓ Req. 10 | ✓ Art. 12 | ✓ A.6.3 |
| Encryption at rest | ✓ CC6.1 | ✓ §164.312(a)(2)(iv) | ✓ Art. 32 | ✓ Req. 3 | implicit | ✓ A.6.5 |
| Data classification | ✓ CC6.5 | ✓ §164.530(c) | ✓ Art. 30 | ✓ Req. 3 | ✓ Art. 10 | ✓ A.5.2 |
| Retention/deletion | ✓ CC6.5 | ✓ §164.530(j) | ✓ Art. 17 | ✓ Req. 3 | — | ✓ A.5.3 |
| Vendor management | ✓ CC9.2 | ✓ §164.314 | ✓ Art. 28 | ✓ Req. 12 | ✓ Art. 28 | ✓ A.7 |
| AI risk assessment | — | — | — | — | ✓ Art. 6–7 | ✓ A.5.1 |
| Bias monitoring | — | — | — | — | ✓ Art. 10 | ✓ A.6.2 |

**Framework-by-framework mechanism summary:**

**SOC 2** — A reporting framework (not a law). An independent auditor (CPA firm) assesses whether your controls match the Trust Service Criteria (Security, Availability, Confidentiality, Processing Integrity, Privacy). Type I = point-in-time. Type II = operating effectiveness over 6–12 months. You ship evidence—logs, screenshots, policy documents—to the auditor. [CITATION NEEDED — concept: SOC 2 Type II evidence format accepted by AICPA]

**HIPAA** — A US federal law. Applies if you create, receive, maintain, or transmit Protected Health Information (PHI). PHI is any individually identifiable health data. Key mechanisms: the Privacy Rule (who can see PHI), the Security Rule (technical safeguards for electronic PHI), and the Breach Notification Rule (what happens when controls fail). If your GTM system touches healthcare prospects and ingests any health-related field, you are in scope.

**GDPR** — An EU regulation. Applies to any entity processing personal data of EU/EEA residents, regardless of where the entity is located. Personal data is any information relating to an identified or identifiable natural person. Key mechanisms: lawful basis for processing (consent, legitimate interest, contract, etc.), data subject rights (access, erasure, portability, objection), Data Processing Impact Assessments (DPIAs) for high-risk processing, and mandatory Data Protection Officer appointment under certain conditions. [CITATION NEEDED — concept: GDPR legitimate interest assessment thresholds for B2B cold outreach]

**PCI-DSS** — An industry standard, not a law. Applies if you store, process, or transmit cardholder data. Cardholder data = primary account number (PAN) plus any of: cardholder name, expiration date, service code. If your GTM stack handles payments or stores card data for any reason, you are in scope. Key mechanism: segmented network architecture that isolates the cardholder data environment (CDE) from everything else. Most GTM systems should architect to never touch card data and delegate to a PCI-compliant payment processor.

**EU AI Act** — An EU regulation (entered force August 2024, phased enforcement). Classifies AI systems by risk: unacceptable (banned), high-risk (mandatory conformity assessment), limited risk (transparency obligations), minimal risk (no regulation). Key mechanism for GTM: AI systems used in employment decisions (hiring, firing, access to self-employment) are classified as high-risk under Annex III. If your AI-powered GTM tool scores, ranks, or filters prospects in ways that affect employment access, you may be in the high-risk category. [CITATION NEEDED — concept: EU AI Act enforcement timeline and specific Annex III categories relevant to sales/marketing AI]

**ISO 42001** — An international standard for AI management systems. Not a law; a certifiable framework. Mirrors the ISO 27001 structure (Plan-Do-Check-Act) but applied to AI-specific risks: bias, explainability, robustness, and human oversight. Mechanism: you define an AI policy, conduct AI risk assessments, implement controls, monitor performance, and undergo third-party certification audit. Currently the only certifiable AI governance standard. [CITATION NEEDED — concept: ISO 42001 certification body availability and audit process]

---

## Beat 3: See It — Observable Compliance Mechanisms in Code

This beat demonstrates three mechanisms as running code: (1) audit logging that produces SOC 2 evidence, (2) GDPR-compliant data deletion with verification, and (3) EU AI Act risk classification of a GTM AI feature.

**Example 1: Structured audit logger for SOC 2 evidence**

```python
import json
import hashlib
from datetime import datetime, timezone

audit_log = []

def log_access(actor, action, resource_type, resource_id, outcome, pii_detected=False):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor_id": hashlib.sha256(actor.encode()).hexdigest()[:16],
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "outcome": outcome,
        "pii_in_scope": pii_detected,
        "control_ref": "SOC2_CC6.1"
    }
    audit_log.append(entry)
    return entry

log_access("sales@example.com", "READ", "prospect_record", "rec-001", "SUCCESS", True)
log_access("api-service", "DELETE", "prospect_record", "rec-002", "SUCCESS", True)
log_access("analyst@example.com", "EXPORT", "campaign_report", "rpt-100", "DENIED", False)

for entry in audit_log:
    print(json.dumps(entry, indent=2))
```

Output confirms each access event is captured with the SOC 2 control reference, a hashed actor identifier, and a PII flag.

**Example 2: GDPR right-to-erasure with verification**

```python
prospect_db = {
    "eu-prospect-001": {
        "email": "maria@example.eu",
        "company": "GmbH Corp",
        "consent_date": "2024-01-15",
        "last_contact": "2024-06-01",
        "phone": "+49-xxx-xxx"
    },
    "us-prospect-002": {
        "email": "john@example.com",
        "company": "LLC Inc",
        "consent_date": None,
        "last_contact": "2024-05-20",
        "phone": "+1-xxx-xxx"
    }
}

erasure_log = []

def execute_erasure(prospect_id, requesting_region):
    if prospect_id not in prospect_db:
        return {"error": f"{prospect_id} not found"}

    record = prospect_db[prospect_id]

    for field in record:
        record[field] = "[REDACTED_PER_GDPR_ART17]"

    prospect_db[prospect_id] = record

    erasure_entry = {
        "prospect_id": prospect_id,
        "erasure_timestamp": datetime.now(timezone.utc).isoformat(),
        "requesting_region": requesting_region,
        "fields_redacted": list(record.keys()),
        "verification_hash": hashlib.sha256(prospect_id.encode()).hexdigest()[:12],
        "retention_override": False
    }
    erasure_log.append(erasure_entry)
    return erasure_entry

result = execute_erasure("eu-prospect-001", "EU")
print("Erasure result:")
print(json.dumps(result, indent=2))

print("\nVerification - record state after erasure:")
print(json.dumps(prospect_db["eu-prospect-001"], indent=2))
```

Output shows the record is redacted field-by-field and the erasure event is logged with a verification hash.

**Example 3: EU AI Act risk tier classification**

```python
def classify_ai_risk(system_name, use_case, affected_persons, human_oversight, bias_testing):
    risk_factors = {
        "employment_related": any(term in use_case.lower() for term in ["hiring", "recruitment", "candidate", "employment", "job"]),
        "affects_eu_citizens": "EU" in affected_persons or "EU/EEA" in affected_persons,
        "automated_decision": "scoring" in use_case.lower() or "ranking" in use_case.lower() or "filtering" in use_case.lower(),
        "has_human_oversight": human_oversight,
        "has_bias_testing": bias_testing
    }

    if risk_factors["employment_related"] and risk_factors["affects_eu_citizens"]:
        tier = "HIGH_RISK"
        obligations = [
            "Conformity assessment required (Art. 6, Annex III)",
            "Risk management system (Art. 9)",
            "Data governance requirements (Art. 10)",
            "Technical documentation (Art. 11)",
            "Transparency to affected persons (Art. 13)",
            "Human oversight measures (Art. 14)",
            "Accuracy, robustness, cybersecurity (Art. 15)"
        ]
    elif risk_factors["automated_decision"] and not risk_factors["has_human_oversight"]:
        tier = "HIGH_RISK"
        obligations = ["Conduct DPIA", "Implement human oversight", "Provide opt-out mechanism"]
    elif "chatbot" in use_case.lower() or "content" in use_case.lower():
        tier = "LIMITED_RISK"
        obligations = ["Transparency obligation: disclose AI interaction (Art. 52)"]
    else:
        tier = "MINIMAL_RISK"
        obligations = ["No mandatory obligations; voluntary best practices"]

    return {
        "system": system_name,
        "risk_tier": tier,
        "risk_factors": risk_factors,
        "obligations": obligations
    }

result = classify_ai_risk(
    "Lead Scoring Engine",
    "AI-based scoring and ranking of sales prospects",
    ["EU/EEA", "US", "UK"],
    human_oversight=False,
    bias_testing=False
)
print(json.dumps(result, indent=2))
```

Output classifies a GTM lead scoring engine as HIGH_RISK under the EU AI Act because it performs automated scoring affecting EU citizens without human oversight.

---

## Beat 4: Use It — GTM Cluster 15 (Security, Auth, Compliance)

**GTM cluster context:** Your GTM stack has an attack surface—rotating API keys, securing webhooks, handling prospect data under GDPR. Compliance is not a separate concern; it constrains every GTM data flow.

**Specific applications:**

**GDPR in outbound sequences:** Every prospect record in your CRM that originated from an EU/EEA source requires a documented lawful basis. If you use legitimate interest, you must maintain a Legitimate Interest Assessment (LIA) document and honor opt-outs within 72 hours. Your sequence engine must check a `consent_status` field before enqueueing a contact. Mechanism: a pre-send filter that queries the consent status and drops or defers non-consented EU records.

```python
prospects = [
    {"id": "p-001", "email": "anna@de-company.eu", "region": "EU", "consent": "legitimate_interest", "opt_out": False},
    {"id": "p-002", "email": "bob@uk-company.uk", "region": "UK", "consent": None, "opt_out": False},
    {"id": "p-003", "email": "carol@us-company.com", "region": "US", "consent": None, "opt_out": False},
    {"id": "p-004", "email": "dirk@nl-company.eu", "region": "EU", "consent": "consent", "opt_out": True},
]

def gdpr_send_filter(prospect_list):
    approved = []
    blocked = []
    for p in prospect_list:
        if p["region"] == "EU":
            if p["opt_out"]:
                blocked.append({"id": p["id"], "reason": "GDPR_OPT_OUT_ACTIVE"})
            elif p["consent"] not in ["consent", "legitimate_interest"]:
                blocked.append({"id": p["id"], "reason": "GDPR_NO_LAWFUL_BASIS"})
            else:
                approved.append(p)
        else:
            if p["opt_out"]:
                blocked.append({"id": p["id"], "reason": "OPT_OUT_ACTIVE"})
            else:
                approved.append(p)
    return {"approved": approved, "blocked": blocked}

result = gdpr_send_filter(prospects)
print("Approved to send:")
for p in result["approved"]:
    print(f"  {p['id']}: {p['email']}")
print("\nBlocked:")
for b in result["blocked"]:
    print(f"  {b['id']}: {b['reason']}")
```

**SOC 2 evidence for webhook integrations:** Every Clay webhook call that reads or writes prospect data is a SOC 2-auditable event. Log the actor, action, resource, and outcome. Your auditor will sample these logs during Type II review. The audit logging code from Beat 3 becomes a function you wrap around every GTM API call.

**PCI-DSS scoping for payment-linked GTM:** If your GTM system triggers invoicing or stores payment references, determine whether cardholder data touches your infrastructure. If it does, scope the CDE boundary. If it does not (you use Stripe/Polaris and never see the PAN), document that decision as evidence of scoping exclusion.

**EU AI Act for AI-generated outreach:** If you use an LLM to generate personalized outreach at scale, classify the system. Personalized cold email generation is likely LIMITED_RISK (transparency obligation: you may need to disclose AI-generated content). If the AI scores or ranks prospects for employment-related opportunities, it may be HIGH_RISK. The classification code from Beat 3 becomes a function you run for every AI feature in your GTM stack.

---

## Beat 5: Build It — Exercises

**Easy:** Extend the audit logger from Beat 3 to accept a `framework_tag` parameter that maps each log entry to all applicable frameworks (e.g., a single access event tagged as both `SOC2_CC6.1` and `GDPR_Art30`). Print the merged log and count entries per framework.

**Medium:** Build a data inventory scanner that takes a list of GTM system fields (email, company, phone, health_insurance_provider, payment_card_last_four) and classifies each field by which frameworks govern it (HIPAA for health data, PCI-DSS for card data, GDPR for personal data). Print a compliance matrix showing field → frameworks.

**Hard:** Implement a compliance-as-code pipeline that (1) reads a mock GTM data flow configuration (data sources, processors, storage locations, third-party vendors), (2) checks each flow against GDPR data processing requirements (lawful basis documented, DPA with vendor, data minimized), (3) flags violations, and (4) outputs a machine-readable compliance report. Include a test case where a vendor without a DPA processes EU prospect data—confirm the pipeline catches it.

---

## Beat 6: Ship It — Compliance Evidence as a Living Artifact

Compliance is not a quarterly audit event. It is a continuous evidence-generation system that runs alongside your GTM stack.

**Integration point:** Add the audit logger and GDPR filter from this lesson to your project codebase. Wire the audit logger into every function that reads or writes prospect data. Wire the GDPR filter into every outbound sequence. Wire the AI risk classifier into your feature registration system so that every new AI-powered GTM feature is classified before launch.

**Evidence output:** Your compliance pipeline should produce three artifacts that an auditor or regulator can request at any time:

1. **Access log export** — structured JSON showing who accessed what, when, with what outcome, tagged by control reference.
2. **Data inventory** — a living document listing every data field in your GTM system, its classification, applicable frameworks, retention policy, and deletion mechanism.
3. **AI system register** — a list of every AI system in your GTM stack, its EU AI Act risk tier, required obligations, and current compliance status.

**GTM redirect (Cluster 15):** Your GTM stack has an attack surface—rotating API keys, securing webhooks, handling prospect data under GDPR. The controls you built in this lesson are the defense layer for Zone 15. Ship the audit logger into your webhook handlers. Ship the GDPR filter into your sequence engine. Ship the AI classifier into your feature flags. When your next enterprise prospect sends a 200-question security questionnaire, your compliance pipeline generates the evidence instead of your team manually screenshotting dashboards.