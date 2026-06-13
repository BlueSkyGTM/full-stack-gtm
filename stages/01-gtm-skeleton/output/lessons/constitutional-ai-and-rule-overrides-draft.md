# Constitutional AI and Rule Overrides

## Beat 1: Hook

You've asked an LLM to score a lead and it returned "10/10" for a company that sells ice to eskimos. The model isn't wrong — you just didn't tell it what "good" means. Constitutional AI is the pattern of encoding your evaluation criteria as explicit rules, then forcing the model to critique its own output against those rules before returning it. Without this, every prompt is a coin flip with a system prompt.

## Beat 2: Concept

Constitutional AI (CAI) is a self-correction loop: generate → critique against rules → revise. Originally developed for safety alignment at Anthropic, the mechanism is portable — any set of rules can serve as the "constitution." The practitioner writes the rules, the model evaluates itself against them, and the revised output is what you actually use. Rule overrides are the escape hatch: explicit conditions where the constitution does not apply or where a higher-priority rule supersedes a lower one.

## Beat 3: Mechanism

The CAI loop has three passes. First pass: the model generates a candidate response. Second pass: the model receives the candidate plus the constitution and produces a critique identifying violations. Third pass: the model revises the candidate using the critique. Each rule in the constitution has a condition, a constraint, and a priority. Rule overrides use priority ordering: if rule A (priority 1) conflicts with rule B (priority 3), rule A wins. This is deterministic resolution, not fuzzy interpretation. The critique pass is where the model acts as evaluator — it must cite which rule was violated and how.

**Exercise hooks:**
- Easy: Write a 3-rule constitution for ICP scoring and run the CAI loop on a sample company profile. Print the original, critique, and revised outputs.
- Medium: Add rule overrides with priority levels. Show that a priority-1 override suppresses a priority-3 rule in the revision.
- Hard: Implement a multi-pass CAI loop that re-critiques its own revision until zero violations remain or 3 iterations elapse. Print violation counts per iteration.

## Beat 4: Use It

**GTM Cluster: Zone 1 — ICP Definition & Scoring**

This is the Clay scoring waterfall with constitutional guardrails. When you build a lead scoring formula in Clay, you're writing rules. CAI gives you a mechanism to enforce those rules at inference time — the model checks its own score against your ICP criteria before returning it. Rule overrides map directly to tier exceptions: "score as qualified if revenue > $50M regardless of employee count" is a priority override. The constitution is your ICP definition document, codified as structured rules.

**Exercise hooks:**
- Easy: Encode 4 ICP criteria as constitutional rules and score 5 sample companies through the CAI loop. Print which rules each company violated.
- Medium: Build a priority override system where "Fortune 500" status bypasses all other qualification rules. Demonstrate with a company that fails 3 of 4 criteria but triggers the override.
- Hard: Chain CAI output into a Clay-style waterfall: enrich → score with constitution → override check → final disposition. Print each stage.

## Beat 5: Code It

Implement the full CAI loop in Python using the Anthropic API. The constitution is a list of dictionaries with `rule_id`, `condition`, `constraint`, and `priority` fields. The function takes a prompt and a constitution, runs generate-critique-revise, and returns the final output plus a violation log. Rule overrides are implemented by sorting rules by priority and short-circuiting when a higher-priority rule resolves a conflict.

**Exercise hooks:**
- Easy: Build and run a single CAI pass with 3 rules. Print the violation log.
- Medium: Implement iterative CAI (re-critique until clean or max iterations). Print iteration count and remaining violations.
- Hard: Build a rule conflict detector that pre-validates the constitution for contradictory rules before any API call. Flag pairs of rules that could conflict given overlapping conditions.

## Beat 6: Ship It

Production CAI has two failure modes: the critique pass misses violations (false negative on rule adherence) and the revision pass introduces new violations (regression). Mitigate with structured output: force the critique into a JSON schema with `violation_found: bool`, `rule_id: str`, `explanation: str`. Log every critique-revise cycle. Set a maximum revision limit (3 is standard) — if violations persist after 3 revisions, escalate to human review. Rule overrides in production need a dry-run mode: run the constitution against historical data and compare override trigger rates against expected rates before deploying.

**Exercise hooks:**
- Easy: Add JSON-structured critique output to your CAI loop. Parse and validate the schema. Print any malformed critiques.
- Medium: Build a violation logger that writes each CAI cycle to a JSONL file with timestamp, rule violations, and revision number. Run 10 scoring calls and read back the log.
- Hard: Implement dry-run mode. Feed 50 historical company profiles through the constitution, collect violation rates per rule, and flag any rule that triggers on >80% or <5% of inputs (indicating the rule is either too strict or useless). Print the audit report.

---

**Learning Objectives (for `docs/en.md`):**
1. Implement a generate-critique-revise loop using a rule-based constitution.
2. Configure priority-based rule overrides to resolve conflicting constraints.
3. Evaluate CAI output quality by tracking violation counts across iterations.
4. Build structured logging for CAI critique cycles in a production context.
5. Detect overbroad or contradictory rules using dry-run validation against historical data.