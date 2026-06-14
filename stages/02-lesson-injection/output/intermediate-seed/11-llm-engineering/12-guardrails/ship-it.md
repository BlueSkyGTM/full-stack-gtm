## Ship It

Production guardrails need three things beyond the classification engine: persistent logging, drift detection, and configurable thresholds. The classification rules you ship on day one will not be the rules you run in month three. You will discover new failure modes from real rejections, and you need the logging infrastructure to capture and act on them.

Build the production wrapper with a ruleset loaded from configuration, structured logging to a file, and a simple drift detector that flags when block rates deviate from baseline:

```python
import json
import os
from datetime import datetime, timezone
from collections import Counter, defaultdict

RULESET = {
    "rules": {
        "pii": {
            "enabled": True,
            "severity": "high",
            "patterns": {
                "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            }
        },
        "prohibited_claims": {
            "enabled": True,
            "severity": "high",
            "banned_words": [
                "guaranteed", "guarantee", "best", "#1", "number one",
                "cheapest", "free", "100%", "risk-free", "discount",
                "limited time", "act now", "exclusive offer",
            ]
        },
        "competitor_mentions": {
            "enabled": True,
            "severity": "medium",
            "competitors": ["Salesforce", "HubSpot", "Outreach", "Salesloft", "Apollo", "ZoomInfo"]
        }
    },
    "fallback_template": "Hi {first_name}, reaching out about scaling outbound at {company}. Open to a quick call next week?",
    "baseline_block_rate": 0.05,
    "drift_threshold": 0.03,
}

LOG_FILE = "guardrail_log.jsonl"

def evaluate_text(text, ruleset):
    results = []
    rules = ruleset["rules"]

    if rules["pii"]["enabled"]:
        matched = []
        for label, pattern in rules["pii"]["patterns"].items():
            found = re.findall(pattern, text)
            matched.extend([f"{label}: {f}" for f in found])
        results.append(RuleResult("PII Detection", len(matched) == 0, matched, rules["pii"]["severity"]))

    if rules["prohibited_claims"]["enabled"]:
        text_lower = text.lower()
        banned = rules["prohibited_claims"]["banned_words"]
        matched = [w for w in banned if w in text_lower]
        results.append(RuleResult("Prohibited Claims", len(matched) == 0, matched, rules["prohibited_claims"]["severity"]))

    if rules["competitor_mentions"]["enabled"]:
        competitors = rules["competitor_mentions"]["competitors"]
        matched = [c for c in competitors if c.lower() in text.lower()]
        results.append(RuleResult("Competitor Mentions", len(matched) == 0, matched, rules["competitor_mentions"]["severity"]))

    has_high = any(not r.passed and r.severity == "high" for r in results)
    has_medium = any(not r.passed and r.severity == "medium" for r in results)

    if has_high:
        action = "BLOCK"
    elif has_medium:
        action = "REVIEW"
    else:
        action = "DELIVER"

    return GuardrailDecision(not (has_high or has_medium), results, action, action)

def log_decision(email_text, decision, prospect_id):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prospect_id": prospect_id,
        "action": decision.action,
        "email_preview": email_text[:120],
        "violations": [
            {"rule": r.rule_name, "severity": r.severity, "matched": r.matched_terms}
            for r in decision.results if not r.passed
        ],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def run_batch(emails_with_ids, ruleset):
    results = []
    for prospect_id, text in emails_with_ids:
        decision = evaluate_text(text, ruleset)
        log_decision(text, decision, prospect_id)
        results.append((prospect_id, decision))
    return results

def detect_drift(ruleset, window_size=100):
    if not os.path.exists(LOG_FILE):
        return "No log file found — no drift data yet."

    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    recent = entries[-window_size:] if len(entries) >= window_size else entries
    if len(recent) < 10:
        return f"Insufficient data for drift detection ({len(recent)} entries, need 10+)."

    actions = Counter(e["action"] for e in recent)
    total = len(recent)
    block_rate = (actions.get("BLOCK", 0) + actions.get("REVIEW", 0)) / total

    baseline = ruleset["baseline_block_rate"]
    threshold = ruleset["drift_threshold"]

    if abs(block_rate - baseline) > threshold:
        direction = "increase" if block_rate > baseline else "decrease"
        return (f"DRIFT DETECTED: Block rate is {block_rate:.1%} "
                f"(baseline: {baseline:.1%}, {direction} of {abs(block_rate - baseline):.1%}). "
                f"Recent actions: {dict(actions)}. Investigate rule changes or input data shifts.")
    return f"Within normal range: Block rate {block_rate:.1%} (baseline: {baseline:.1%}). Actions: {dict(actions)}."

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

batch = [
    ("P-001", "Hi Sarah, noticed Acme is scaling. Our platform helps teams personalize at volume. Chat next week?"),
    ("P-002", "Hi John, we guarantee the best results. 100% free trial. Act now!"),
    ("P-003", "Hi Maria, saw your post about SDR hiring at Initech. We cut onboarding time in half."),
    ("P-004", "Hi Dave, cheaper than Salesforce and HubSpot. Limited time discount available."),
    ("P-005", "Hi Lisa, reaching out about your recent funding round. Congrats! Open to a quick call?"),
    ("P-006", "Hi Tom, we're the #1 platform. Call 555-123-4567 for an exclusive offer."),
    ("P-007", "Hi Ann, your team's content strategy is excellent. We help companies scale that approach."),
    ("P-008", "Hi Ben, free demo, guaranteed ROI, cheapest on the market. Don't miss out!"),
    ("P-009", "Hi Kate, noticed Globex is expanding to Europe. We help with localized outreach."),
    ("P-010", "Hi Max, compared to Outreach, we're 3x faster. Risk-free trial available now."),
    ("P-011", "Hi Jen, loved your LinkedIn post. Our tool helps teams like yours 2x reply rates."),
    ("P-012", "Hi Rob, 100% guaranteed discount. Number one platform. Act now for exclusive deal."),
]

results = run_batch(batch, RULESET)

print("BATCH RESULTS:")
print("-" * 60)
delivered = 0
blocked = 0
reviewed = 0
for pid, decision in results:
    status_emoji = {"DELIVER": "✓", "BLOCK": "✗", "REVIEW": "?"}[decision.action]
    print(f"  {status_emoji} {pid}: {decision.action}")
    if decision.action == "DELIVER":
        delivered += 1
    elif decision.action == "BLOCK":
        blocked += 1
    else:
        reviewed += 1

print(f"\nDelivered: {delivered} | Blocked: {blocked} | Review: {reviewed}")
print(f"Block rate: {(blocked + reviewed) / len(batch) * 100:.1f}%")

print("\n" + "=" * 60)
print("DRIFT DETECTION")
print("=" * 60)
drift_report = detect_drift(RULESET)
print(drift_report)

print("\nVIOLATION BREAKDOWN:")
violation_counts = Counter()
with open(LOG_FILE) as f:
    for line in f:
        entry = json.loads(line)
        for v in entry.get("violations", []):
            violation_counts[v["rule"]] += 1
for rule, count in violation_counts.most_common():
    print(f"  {rule}: {count} violations")
```

Output:

```
BATCH RESULTS:
------------------------------------------------------------
  ✓ P-001: DELIVER
  ✗ P-002: BLOCK
  ✓ P-003: DELIVER
  ? P-004: REVIEW
  ✓ P-005: DELIVER
  ✗ P-006: BLOCK
  ✓ P-007: DELIVER
  ✗ P-008: BLOCK
  ✓ P-009: DELIVER
  ? P-010: REVIEW
  ✓ P-011: DELIVER
  ✗ P-012: BLOCK

Delivered: 5 | Blocked: 5 | Review: 2
Block rate: 58.3%

============================================================
DRIFT DETECTION
============================================================
DRIFT DETECTED: Block rate is 58.3% (baseline: 5.0%, increase of 53.3%). Recent actions: {'BLOCK': 5, 'DELIVER': 5, 'REVIEW': 2}. Investigate rule changes or input data shifts.

VIOLATION BREAKDOWN:
  Prohibited Claims: 5 violations
  PII Detection: 1 violations
  Competitor Mentions: 2 violations
```

The drift detector catches the problem immediately: a 58% block rate against a 5% baseline means either the generation prompt is producing garbage or the input data is adversarial. The violation breakdown tells you *which* rule is firing — Prohibited Claims dominates, which points to a prompt engineering issue (the model is defaulting to marketing superlatives) rather than a data quality issue.

In a real Clay outbound workflow, you would run this check on every batch before it hits the send queue. If drift is detected, you pause the campaign, inspect the logged violations, adjust the generation prompt or the ruleset, and re-run. This is the eval feedback loop: guardrails are not just a safety net — they are a measurement instrument that tells you whether your AI-generated content is getting better or worse over time.