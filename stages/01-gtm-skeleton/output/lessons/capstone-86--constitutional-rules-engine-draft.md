# Capstone 86 — Constitutional Rules Engine

## Hook

You've built prompts, chained agents, and debugged hallucinations. Now you need a system that enforces policy — not as a system prompt hoping for compliance, but as executable logic that blocks violations before they reach the user. This capstone pulls together rule representation, evaluation order, and override mechanics into one running engine.

---

## Concept

**Separation of constitution from interpretation.** A rules engine works by encoding constraints as data (rule objects with conditions, actions, and priority) rather than embedding them in prompt text. The engine evaluates inputs against every rule in priority order, short-circuiting on blocks, and logging passes. This is the same pattern behind content-moderation APIs and compliance gates — the constitution is a list, the interpreter is a loop, and the output is a verdict plus an audit trail.

Key mechanisms:
- **Rule schema**: Each rule has an ID, a condition function, a severity (block | warn | pass), an explanation template, and a priority integer.
- **Evaluation loop**: Iterate rules in priority order; return immediately on first `block`; accumulate `warn` reasons; return `pass` with all accumulated warnings if no block fires.
- **Override protocol**: Authorized users can mark a specific rule as bypassed for a specific input, persisted in a log with a reason string.
- **Audit trail**: Every evaluation writes input hash, rule ID, verdict, timestamp, and override flag to a structured log.

---

## Demonstration

Build a minimal constitutional rules engine in Python that:
1. Defines 5 rules as data objects (e.g., "no PII in output", "no pricing claims without source", "no competitor comparisons without approval", "tone check for professional language", "length under 500 words").
2. Runs 3 test inputs through the engine — one clean, one that triggers a warn, one that triggers a block.
3. Prints the verdict, reasons, and audit log for each input.

All code runs in terminal with observable print output. No external APIs, no browser dependency.

---

## Use It

**GTM Redirect → Zone 1 Outbound / ICP Qualification Rules**

This is the same architecture behind Clay's waterfall enrichment and Smartlead's delivery rules: policy-as-data, evaluated in order, with audit trails. In outbound ops, your ICP qualification criteria are a constitution — firmographic rules, intent-signal rules, exclusion rules — and your lead-routing logic is the evaluation engine. The capstone's rule schema maps directly to a Clay enrichment waterfall where each column is a condition check, and the final verdict determines whether a lead enters a sequence or gets recycled.

[CITATION NEEDED — concept: Clay waterfall architecture rule-evaluation order]

---

## Ship It

Exercise hooks (full specs not written here):

- **Easy**: Add a sixth rule to the engine that checks for a specific keyword and test it against two inputs. Print the updated verdicts.
- **Medium**: Implement the override protocol — accept a JSON payload with `rule_id`, `input_hash`, `override_reason`, and `authorized_by`. Re-run the blocked input with the override active. Confirm the verdict changes and the override appears in the audit trail.
- **Hard**: Implement rule conflict detection — when two rules have the same priority and contradict (one blocks, one allows), the engine must surface a `conflict` verdict with both rule IDs and refuse to evaluate until the conflict is resolved in the rule data.

---

## Assess It

Quiz hooks (full questions not written here):

1. **Mechanism question**: Given a rules engine that evaluates in priority order and short-circuits on block, trace the evaluation path for a specific input against a provided rule set. Identify which rules fire and which are skipped.
2. **Design question**: Explain why encoding rules as data objects rather than prompt instructions produces more reliable enforcement. Identify one tradeoff.
3. **Debugging question**: Given a rule that always passes even when it should block, identify the most likely bug in the condition function and propose the fix.
4. **GTM mapping question**: Map three ICP qualification criteria (e.g., company size > 50, industry = SaaS, funding stage > Seed) to the rule schema from the engine. Identify which should be a block vs. a warn.

---

**Learning Objectives (testable, action verbs):**

1. Implement a rules engine that evaluates inputs against an ordered list of rule objects and returns structured verdicts.
2. Distinguish between block, warn, and pass severities and trace evaluation short-circuit behavior.
3. Build an override protocol that bypasses a specific rule for a specific input with an audit trail.
4. Map GTM qualification criteria to the rule schema pattern and explain the mapping.