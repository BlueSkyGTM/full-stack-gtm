# Security — Secrets, API Key Rotation, Audit Logs, Guardrails

## Learning Objectives

1. Detect leaked secrets in text and files using regex pattern matching and entropy analysis
2. Implement time-based API key rotation with zero-downtime key swapping
3. Write structured audit log entries that support replay, forensics, and compliance review
4. Build input/output guardrails that reject content matching defined patterns
5. Compare rotation strategies by their trade-offs between security and availability

---

## Hook It

A single API key committed to a public repo can be scraped and exploited within 90 seconds. When your GTM automation holds production credentials for Clay, SendGrid, and Salesforce, that attack surface is not theoretical — it's a Friday afternoon incident waiting to happen. This lesson covers the four mechanisms that prevent that incident: secret detection, key rotation, audit logging, and guardrails.

---

## Ground It

Secrets in GTM workflows live in environment variables, `.env` files, and platform credential stores. Review how credentials flow through the Clay waterfall, how Make/n8n store API keys, and where the exposure points are when an agent runs autonomously. This beat maps the attack surface: git history, log output, error messages, and agent memory.

**Exercise hook (easy):** Given a mock `.env` file and a git diff, identify every location where a secret could leak.

---

## Build It

Four mechanisms, implemented in order:

**Secret scanning** — regex patterns for common key formats (AWS `AKIA...`, Stripe `sk_live_...`, generic high-entropy strings) applied to file content and stdout.

**Key rotation** — a dual-key pattern where old and new keys coexist during a rotation window, then the old key is revoked. Implemented as a scheduled function that swaps credentials in environment variables.

**Audit logging** — structured JSON entries with timestamp, actor, action, resource, and a SHA-256 checksum. Every mutation your agent performs writes a log entry before execution.

**Guardrails** — regex and keyword filters applied to LLM input (prevent prompt injection) and output (prevent PII leakage, unauthorized domain references). The guardrail function wraps the LLM call and returns a rejection or the clean response.

```python
import re
import hashlib
import json
import os
from datetime import datetime, timezone

SECRET_PATTERNS = [
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
    (r'sk_live_[0-9a-zA-Z]{24,}', 'Stripe Live Key'),
    (r'glpat-[0-9a-zA-Z\-]{20,}', 'GitLab PAT'),
]

def scan_for_secrets(text):
    findings = []
    for pattern, label in SECRET_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            findings.append({"type": label, "value": match[:8] + "..." + match[-4:]})
    return findings

def write_audit_entry(actor, action, resource, detail=""):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "resource": resource,
        "detail": detail
    }
    entry_str = json.dumps(entry, sort_keys=True)
    entry["checksum"] = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
    print(json.dumps(entry, indent=2))
    return entry

def apply_guardrail(text, blocked_patterns=None):
    if blocked_patterns is None:
        blocked_patterns = [r'competitor\.com', r'password\s*=', r'ssn']
    for pattern in blocked_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"allowed": False, "reason": f"Matched blocked pattern: {pattern}"}
    return {"allowed": True, "reason": None}

sample_text = "My AWS key is AKIAIOSFODNN7EXAMPLE and my Stripe key is sk_EXAMPLE_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
leaks = scan_for_secrets(sample_text)
print("Secret scan results:")
for leak in leaks:
    print(f"  - {leak['type']}: {leak['value']}")

print("\nAudit log entry:")
write_audit_entry("agent-01", "enrich", "clay-waterfall", "Rotated API key post-enrichment")

print("\nGuardrail test:")
print(apply_guardrail("Send email to sales@competitor.com"))
print(apply_guardrail("Send email to sales@example.com"))
```

**Exercise hook (medium):** Build a pre-commit hook script that runs `scan_for_secrets` on staged files and blocks the commit if findings exist.

---

## Use It

**GTM redirect:** This maps to Zone 03 (Enrich & Qualify) and Zone 04 (Outreach & Engagement) in the GTM topic map — specifically the Clay waterfall and autonomous outreach sequences.

In GTM automation, security failures have direct revenue consequences:

- **Key rotation in Clay:** When you rotate the Clay API key, every running waterfall that references it must complete before revocation. The dual-key pattern lets you deploy a new key without killing in-flight enrichments. [CITATION NEEDED — concept: Clay API key rotation workflow]

- **Audit logs for outreach compliance:** When an agent sends 500 emails autonomously, you need a forensic record of *what was sent, to whom, by which agent version, at what time*. This is the audit log. Without it, you cannot debug a spam complaint or prove opt-in compliance.

- **Guardrails on outbound content:** An LLM writing outreach emails might reference a competitor by name, include pricing you didn't approve, or hallucinate a feature. Output guardrails catch this before the email hits SendGrid.

**Exercise hook (medium):** Given a mock Clay webhook payload, apply guardrails that reject any payload containing competitor domains or unapproved pricing references, and log every rejection as an audit entry.

---

## Ship It

Production checklist for securing a GTM agent system:

1. **Secrets management:** All API keys in `.env`, `.env` in `.gitignore`, pre-commit hook runs `scan_for_secrets` on every commit
2. **Rotation schedule:** Cron job or CI step that generates new keys, updates `.env`, and triggers a grace-period timer for old-key revocation
3. **Audit log destination:** JSON lines file rotated daily, or a SQLite database for queryable forensics
4. **Guardrail middleware:** A function that wraps every LLM call — both the prompt (input) and the completion (output) pass through the guardrail filter

The ship artifact is a security layer that can be imported into any agent script:

```python
import re
import hashlib
import json
import os
from datetime import datetime, timezone

SECRET_PATTERNS = [
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
    (r'sk_live_[0-9a-zA-Z]{24,}', 'Stripe Live Key'),
    (r'glpat-[0-9a-zA-Z\-]{20,}', 'GitLab PAT'),
]

BLOCKED_OUTPUT_PATTERNS = [
    r'competitor\.com',
    r'\$[\d,]+(?:/mo|/month|per month)',
    r'password\s*[:=]',
]

class SecurityLayer:
    def __init__(self, audit_log_path="audit.jsonl"):
        self.audit_log_path = audit_log_path

    def scan_text(self, text, source="unknown"):
        findings = []
        for pattern, label in SECRET_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                findings.append({"type": label, "masked": match[:6] + "..." + match[-4:], "source": source})
        if findings:
            self._log("secret-detected", source, json.dumps(findings))
        return findings

    def guard_output(self, text, source="llm-output"):
        for pattern in BLOCKED_OUTPUT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self._log("guardrail-blocked", source, f"Pattern: {pattern}, Match: {match.group()}")
                return {"allowed": False, "reason": f"Blocked pattern: {pattern}", "match": match.group()}
        self._log("guardrail-passed", source, "Output clean")
        return {"allowed": True, "reason": None, "match": None}

    def _log(self, action, actor, detail):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "detail": detail
        }
        entry_str = json.dumps(entry, sort_keys=True)
        checksum = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
        entry["checksum"] = checksum
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

sec = SecurityLayer("audit.jsonl")

test_input = "Deploy with key AKIAIOSFODNN7EXAMPLE and Stripe sk_EXAMPLE_xxxxxxxxxxxxxxxxxxxx1234"
scan_results = sec.scan_text(test_input, "agent-config-file")
print(f"Scan found {len(scan_results)} secrets:")
for s in scan_results:
    print(f"  {s['type']}: {s['masked']}")

email_draft = "Hi, our product costs $499/month and integrates with competitor.com tools."
guard_result = sec.guard_output(email_draft, "outreach-agent")
print(f"\nGuardrail result: allowed={guard_result['allowed']}, reason={guard_result['reason']}")

clean_draft = "Hi, I'd love to show you how our platform can help your team."
guard_result_clean = sec.guard_output(clean_draft, "outreach-agent")
print(f"Guardrail result: allowed={guard_result_clean['allowed']}")

print("\nAudit log contents:")
with open("audit.jsonl", "r") as f:
    for line in f:
        print(line.strip())

os.remove("audit.jsonl")
```

**Exercise hook (hard):** Extend `SecurityLayer` with a `rotate_key` method that accepts a key name, generates a new value, writes it to `.env`, and sets a revocation timer for the old key — all while logging every step to the audit trail.

---

## Extend It

- **Dynamic secrets:** HashiCorp Vault generates short-lived credentials that expire after a TTL — no rotation needed because keys die on schedule
- **OAuth2 service accounts:** Eliminate API keys entirely by using service-to-service authentication with signed JWTs
- **Content moderation APIs:** OpenAI's moderation endpoint and similar tools provide production-grade guardrails beyond regex — use them for output validation on user-facing content
- **Git secret scanning:** GitHub's built-in push protection and secret scanning run server-side as a backup to your pre-commit hooks
- **Agent sandboxing:** Run autonomous agents in containers with network policies that restrict which domains they can reach — defense in depth beyond key management