## Ship It

**Easy:** The four scenarios above are your test suite. Add a fifth scenario that triggers loop exhaustion: an agent that calls `lookup_company` on each step, gets `"incomplete"` in its thought, and never converges. Run the diagnostics and confirm `LOOP_EXHAUSTION` is detected.

```python
def run_scenario_loop_exhaustion() -> AgentRun:
    run = AgentRun(max_steps=5)
    for i in range(5):
        step = StepTrace(
            step_num=i + 1,
            thought="Result is incomplete, need to look up more",
            tool_name="lookup_company",
            tool_args={"company_name": f"Company{i}"},
            tool_result=f"company_name: Company{i}",
            context_tokens=200 * (i + 1),
        )
        step.errors = validate_tool_call("lookup_company", step.tool_args, TOOL_SCHEMA)
        run.add_step(step)
    return run

print("\n" + "=" * 70)
print("SCENARIO 5: Loop Exhaustion")
print("=" * 70)
diag5 = diagnose(run_scenario_loop_exhaustion())
print(json.dumps(diag5, indent=2))
```

**Medium:** Add rate limit handling as a seventh detection category. Write `detect_rate_limit_hit` that scans tool results for rate-limit signatures (HTTP 429 patterns, "rate limit" strings, or repeated identical error responses across consecutive steps). Test it by injecting a scenario where steps 3-5 all return `"error: rate_limit_exceeded"`.

```python
def detect_rate_limit_hit(run: AgentRun) -> str | None:
    rate_limit_pattern = "rate_limit"
    consecutive_hits = 0
    first_hit_step = None
    for step in run.steps:
        if step.tool_result and rate_limit_pattern in step.tool_result.lower():
            if consecutive_hits == 0:
                first_hit_step = step.step_num
            consecutive_hits += 1
        else:
            consecutive_hits = 0
            first_hit_step = None
        if consecutive_hits >= 2:
            return (
                f"RATE_LIMIT_HIT starting at step {first_hit_step}: "
                f"{consecutive_hits} consecutive rate-limited responses. "
                f"Agent should back off, not retry immediately."
            )
    return None

def run_scenario_rate_limit() -> AgentRun:
    run = AgentRun()
    steps_data = [
        ("Looking up CorpA", "lookup_company",
         {"company_name": "CorpA"}, "company_name: CorpA", 300),
        ("Looking up CorpB", "lookup_company",
         {"company_name": "CorpB"}, "error: rate_limit_exceeded", 500),
        ("Retrying CorpB", "lookup_company",
         {"company_name": "CorpB"}, "error: rate_limit_exceeded", 600),
        ("Retrying CorpB again", "lookup_company",
         {"company_name": "CorpB"}, "error: rate_limit_exceeded", 700),
    ]
    for i, (thought, tool, args, result, tokens) in enumerate(steps_data):
        errors = validate_tool_call(tool, args, TOOL_SCHEMA) if tool else []
        run.add_step(StepTrace(i + 1, thought, tool, args, result, tokens, errors))
    return run

print("\n" + "=" * 70)
print("SCENARIO 6: Rate Limit (Medium Exercise)")
print("=" * 70)
rl_run = run_scenario_rate_limit()
rl_result = detect_rate_limit_hit(rl_run)
print(rl_result if rl_result else "NOT_DETECTED")
```

**Hard:** Build a recovery mechanism. When `detect_tool_parameter_drift` trips, the agent re-reads the tool schema and retries the call with corrected parameters. Demonstrate recovery: inject a drift failure, detect it, reconstruct the correct args from the schema, and re-run the step successfully.

```python
def recover_from_drift(
    step: StepTrace, schema: dict, prior_valid_args: dict | None = None
) -> dict[str, Any]:
    if not step.tool_name or step.tool_name not in schema:
        return {"recovered": False, "reason": "cannot recover: unknown tool"}
    spec = schema[step.tool_name]
    recovered_args = {}
    for req in spec["required"]:
        if req in step.tool_args and step.tool_args[req] is not None:
            recovered_args[req] = step.tool_args[req]
        elif prior_valid_args and req in prior_valid_args:
            recovered_args[req] = prior_valid_args[req]
        else:
            return {
                "recovered": False,
                "reason": f"cannot recover: no source for required param '{req}'",
            }
    for opt in spec.get("optional", []):
        if opt in step.tool_args and step.tool_args[opt] is not None:
            recovered_args[opt] = step.tool_args[opt]
    expected = spec["param_types"]
    for param, value in recovered_args.items():
        if param in expected and not isinstance(value, eval(expected[param])):
            try:
                if expected[param] == "str":
                    recovered_args[param] = str(value)
                elif expected[param] == "int":
                    recovered_args[param] = int(value)
                elif expected[param] == "float":
                    recovered_args[param] = float(value)
            except (ValueError, TypeError):
                return {
                    "recovered": False,
                    "reason": f"cannot recover: type cast failed for '{param}'",
                }
    return {"recovered": True, "corrected_args": recovered_args}


broken_step = StepTrace(
    step_num=3,
    thought="Finding decision maker",
    tool_name="find_decision_maker",
    tool_args={"company_name": "Acme Corp"},
    tool_result=None,
    context_tokens=900,
)
broken_step.errors = validate_tool_call(
    broken_step.tool_name, broken_step.tool_args, TOOL_SCHEMA
)

prior_valid = {"company_name": "Acme Corp", "title": "CTO"}
recovery = recover_from_drift(broken_step, TOOL_SCHEMA, prior_valid)

print("\n" + "=" * 70)
print("HARD EXERCISE: Recovery from Tool Parameter Drift")
print("=" * 70)
print(f"Broken step errors: {broken_step.errors}")
print(f"Recovery result: {json.dumps(recovery, indent=2)}")
if recovery["recovered"]:
    revalidated = validate_tool_call(
        broken_step.tool_name, recovery["corrected_args"], TOOL_SCHEMA
    )
    print(f"Re-validation after recovery: {revalidated if revalidated else 'PASS — no errors'}")
```