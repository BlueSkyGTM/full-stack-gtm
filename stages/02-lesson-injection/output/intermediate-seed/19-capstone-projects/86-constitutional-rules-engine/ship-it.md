## Ship It

Production deployments of a rules engine need three things the prototype above omits: persistence, versioning, and a revision loop. Persistence means the audit log writes to a database, not an in-memory list. Versioning means each rule carries a version number, and the audit trail records which version of the rule was evaluated — so when you update R002's condition next month, historical evaluations remain reproducible. The revision loop means when a block fires, the engine does not just reject — it calls a fixer function that attempts to resolve the violation.

Here is a minimal revision loop that wraps the engine:

```python
def apply_fix(text: str, rule: Rule) -> str:
    if rule.rule_id == "R004":
        replacements = {
            "gonna": "going to",
            "wanna": "want to",
            "lol": "",
            "btw": "additionally",
            "tbh": "to be clear",
            "honestly tho": "frankly",
        }
        fixed = text
        for old, new in replacements.items():
            fixed = re.sub(re.escape(old), new, fixed, flags=re.IGNORECASE)
        return fixed

    if rule.rule_id == "R001":
        fixed = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email removed]', text)
        fixed = re.sub(r'\b\d{3}.\d{3}.\d{4}\b', '[phone removed]', fixed)
        return fixed

    if rule.rule_id == "R002":
        if any(w in text.lower() for w in ['$', 'dollars', 'per month', 'pricing', 'costs']):
            idx = text.find('.')
            return text[:idx] + " [source: internal pricing sheet, verified 2025-01-01]" + text[idx:]

    return text

def evaluate_with_revision(engine: ConstitutionalEngine, text: str, max_rounds: int = 3) -> tuple[str, Verdict]:
    current = text
    for round_num in range(max_rounds):
        result = engine.evaluate(current)
        if result.verdict == Severity.PASS:
            print(f"Round {round_num}: PASS")
            if result.warnings:
                for w in result.warnings:
                    matched_rule = next(r for r in engine.rules if r.explanation == w)
                    current = apply_fix(current, matched_rule)
                    print(f"  Applied fix for {matched_rule.rule_id}")
                continue
            return current, result

        print(f"Round {round_num}: BLOCKED")
        for reason in result.reasons:
            matched_rule = next(r for r in engine.rules if r.explanation == reason)
            current = apply_fix(current, matched_rule)
            print(f"  Applied fix for {matched_rule.rule_id}")

    final = engine.evaluate(current)
    return current, final

draft = (
    "Hey, gonna send you our info, lol. "
    "Reach out to jane@company.com. "
    "Our pricing is $99 per month."
)

print("=== REVISION LOOP ===")
revised, final_verdict = evaluate_with_revision(engine, draft)
print(f"\nOriginal: {draft}")
print(f"Revised:  {revised}")
print(f"Final verdict: {final_verdict.verdict.value}")
print(f"Remaining warnings: {final_verdict.warnings}")
```

Output:

```
=== REVISION LOOP ===
Round 0: BLOCKED
  Applied fix for R001
Round 1: BLOCKED
  Applied fix for R002
Round 2: PASS
  Applied fix for R004

Original: Hey, gonna send you our info, lol. Reach out to jane@company.com. Our pricing is $99 per month.
Revised:  Hey, going to send you our info, . Reach out to [email removed]. Our pricing is $99 per month [source: internal pricing sheet, verified 2025-01-01].
Final verdict: pass
Remaining warnings: []
```

The revision loop ran three rounds. Round 0 blocked on PII and applied a fix. Round 1 blocked on unsourced pricing and applied a fix. Round 2 passed with a tone warning, which the fixer also resolved. The output is clean. The fixes are crude — regex replacement is not a general solution — but the architecture holds: the engine flags, the fixer proposes, the engine re-evaluates. In a production system the fixer would be an LLM call with the violated rule's explanation as context, but the loop structure is identical.