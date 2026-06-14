## Ship It

The production module wraps the classifier in a clean API with three entry points: `classify_input(text)` for user-facing text entering the pipeline, `classify_output(text)` for model-generated text leaving the pipeline, and `classify_conversation(messages)` for multi-turn exchanges where both sides need scoring. Every `unsafe` verdict logs the category, a hash of the raw text (not the raw text itself, to avoid storing PII in your audit log—this is a GDPR consideration), and a UTC timestamp. A dry-run mode logs verdicts without blocking delivery, which lets you evaluate classifier behavior on real traffic before enforcing policies.

The module below includes a mock outreach generation pipeline that exercises the full loop: generate copy, classify the output, apply the rejection policy, and either deliver or block. The generation step is stubbed with canned responses—you would replace it with your actual LLM call in production. The classification and policy logic is real and observable.

```python
import hashlib
import json
from datetime import datetime, timezone

TAXONOMY = {
    "S1": "Hate Speech",
    "S2": "Harassment",
    "S3": "Violence",
    "S6": "PII Exposure",
    "S7": "Deception/Fraud",
}

VIOLATION_PATTERNS = {
    "S1": ["kill all", "inferior race", "subhuman"],
    "S2": ["you are stupid", "shut up", "worthless"],
    "S3": ["how to harm", "build a weapon", "attack"],
    "S6": ["social security number", "credit card number", "passport number", "bank account"],
    "S7": ["fake identity", "impersonate", "phishing", "ignore previous instructions"],
}

DEFAULT_POLICY = {
    "S1": "block",
    "S2": "human_review",
    "S3": "block",
    "S6": "block",
    "S7": "human_review",
}


class SafetyClassifier:
    def __init__(self, policy=None, dry_run=False):
        self.policy = policy or DEFAULT_POLICY
        self.dry_run = dry_run
        self.audit_log = []

    def _classify(self, text, role):
        violations = []
        text_lower = text.lower()
        for category, patterns in VIOLATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    violations.append(category)
                    break
        return {
            "role": role,
            "safe": len(violations) == 0,
            "violations": violations,
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _apply_policy(self, result):
        if result["safe"]:
            return "deliver"
        actions = [self.policy.get(v, "human_review") for v in result["violations"]]
        if "block" in actions and not self.dry_run:
            return "block"
        if "block" in actions and self.dry_run:
            return "deliver_with_warning"
        return "human_review"

    def _log(self, result, action):
        if not result["safe"]:
            entry = {
                "timestamp": result["timestamp"],
                "role": result["role"],
                "violations": result["violations"],
                "text_hash": result["text_hash"],
                "action": action,
                "dry_run": self.dry_run,
            }
            self.audit_log.append(entry)

    def classify_input(self, text):
        result = self._classify(text, "User")
        action = self._apply_policy(result)
        self._log(result, action)
        return {"classification": result, "action": action}

    def classify_output(self, text):
        result = self._classify(text, "Agent")
        action = self._apply_policy(result)
        self._log(result, action)
        return {"classification": result, "action": action}

    def classify_conversation(self, messages):
        results = []
        for msg in messages:
            if msg["role"] == "user":
                results.append(self.classify_input(msg["content"]))
            else:
                results.append(self.classify_output(msg["content"]))
        return results

    def export_log(self):
        return json.dumps(self.audit_log, indent=2)


MOCK_GENERATED_OUTPUTS = [
    "Hi Sarah, following up on our Q4 platform demo. Would love to reconnect this week.",
    "Hi John, I noticed your bank account was compromised. Click here to verify your identity.",
    "Hey team, here is the weekly update on our product roadmap and customer wins.",
    "To everyone on the list: you are worthless and should shut up about your complaints.",
]


def run_outreach_pipeline(classifier):
    delivered = []
    blocked = []
    reviewed = []

    for output_text in MOCK_GENERATED_OUTPUTS:
        verdict = classifier.classify_output(output_text)
        action = verdict["action"]
        entry = {"text": output_text[:80], "action": action}

        if action == "deliver" or action == "deliver_with_warning":
            delivered.append(entry)
        elif action == "block":
            blocked.append(entry)
        elif action == "human_review":
            reviewed.append(entry)

    return {"delivered": delivered, "blocked": blocked, "reviewed": reviewed}


strict = SafetyClassifier(policy=DEFAULT_POLICY, dry_run=False)
results = run_outreach_pipeline(strict)

print("=== STRICT MODE (enforce) ===")
print(f"Delivered:     {len(results['delivered'])}")
print(f"Blocked:       {len(results['blocked'])}")
print(f"Human Review:  {len(results['reviewed'])}")
for item in results["delivered"]:
    print(f"  ✓ DELIVERED: {item['text']}")
for item in results["blocked"]:
    print(f"  ✗ BLOCKED:   {item['text']}")
for item in results["reviewed"]:
    print(f"  → REVIEW:    {item['text']}")
print()

dry = SafetyClassifier(policy=DEFAULT_POLICY, dry_run=True)
dry_results = run_outreach_pipeline(dry)
print("=== DRY-RUN MODE (log only) ===")
print(f"Delivered (with warnings): {len(dry_results['delivered'])}")
print(f"Blocked:                   {len(dry_results['blocked'])}")
print(f"Human Review:              {len(dry_results['reviewed'])}")
print()

print("=== AUDIT LOG (strict mode) ===")
print(strict.export_log())
```

The output shows the full pipeline behavior. In strict mode, the PII-leaking email and the harassment message get blocked or routed to human review. The two clean outputs deliver normally. In dry-run mode, everything delivers but violations are logged with warnings—useful for shadow-testing the classifier on real traffic before flipping the enforcement switch. The audit log captures every `unsafe` verdict with category, text hash, and timestamp, giving you the compliance trail needed for CAN-SPAM and GDPR audits without storing raw PII in your logs.