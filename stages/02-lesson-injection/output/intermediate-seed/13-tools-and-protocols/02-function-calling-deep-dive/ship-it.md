## Ship It

Production function-calling systems break in predictable ways. Here is how each provider fails and how to handle it.

**Malformed arguments.** OpenAI returns `arguments` as a raw string — `json.loads()` can fail if the model produces invalid JSON. This happens most often with complex nested schemas or when the model is under token pressure. Wrap the parse in a try/except and return the error back to the model as a tool result so it can self-correct:

```python
def safe_parse_arguments(raw_arguments):
    try:
        return json.loads(raw_arguments), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in arguments: {e}. Raw: {raw_arguments[:200]}"

test_cases = [
    '{"company_name": "Clay", "data_field": "headcount"}',
    '{"company_name": "Clay", "data_field": "headcount"',  # broken JSON
    '',
    'null',
]

for tc in test_cases:
    parsed, error = safe_parse_arguments(tc)
    status = "OK" if error is None else "PARSE_ERROR"
    print(f"[{status}] input={tc[:50]!r} -> parsed={parsed}, error={error}")
```

**Parallel function calls.** OpenAI and Gemini both support parallel calls — the model returns multiple tool calls in a single response. Anthropic does too, as multiple `tool_use` blocks in `content[]`. You must execute all of them and inject all results before continuing. Execute them sequentially (or concurrently with `asyncio` if they are I/O-bound), collect results, and append them as separate messages:

```python
def handle_parallel_calls_anthropic(calls):
    results = []
    for call in calls:
        result = execute_tool(call["name"], call["arguments"])
        results.append(anthropic_tool_result(call["id"], result))
    return results

parallel_response = {
    "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "get_company_info",
         "input": {"company_name": "Clay", "data_field": "industry"}},
        {"type": "tool_use", "id": "toolu_2", "name": "get_company_info",
         "input": {"company_name": "Anthropic", "data_field": "headcount"}},
        {"type": "tool_use", "id": "toolu_3", "name": "score_contactability",
         "input": {"has_email": True, "has_phone": True}},
    ]
}

calls = parse_anthropic_response(parallel_response)
print(f"Received {len(calls)} parallel calls from Anthropic")
result_messages = handle_parallel_calls_anthropic(calls)
for i, msg in enumerate(result_messages):
    content = msg["content"][0]["content"]
    print(f"  Result {i+1}: {content}")
```

**Forced tool choice.** Each provider lets you steer whether the model must call a tool, may call a tool, or is forbidden from calling one. OpenAI uses `tool_choice: "auto" | "required" | {"type": "function", "function": {"name": "..."}}`. Anthropic uses `tool_choice: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "..."}`. Gemini uses `tool_config: {function_calling_config: {mode: "AUTO" | "ANY" | "NONE", allowed_function_names: [...]}}`.

```python
def build_tool_choice_openAI(mode="auto", function_name=None):
    if mode == "auto":
        return "auto"
    if mode == "any":
        return "required"
    if mode == "specific" and function_name:
        return {"type": "function", "function": {"name": function_name}}
    raise ValueError(f"Invalid combo: {mode} / {function_name}")

def build_tool_choice_anthropic(mode="auto", function_name=None):
    if mode == "auto":
        return {"type": "auto"}
    if mode == "any":
        return {"type": "any"}
    if mode == "specific" and function_name:
        return {"type": "tool", "name": function_name}
    raise ValueError(f"Invalid combo: {mode} / {function_name}")

def build_tool_choice_gemini(mode="auto", function_names=None):
    mode_map = {"auto": "AUTO", "any": "ANY", "none": "NONE"}
    config = {"mode": mode_map.get(mode, "AUTO")}
    if function_names:
        config["allowed_function_names"] = function_names
    return {"function_calling_config": config}

print("=== Forced tool choice comparison ===\n")
print("OpenAI (specific):", json.dumps(build_tool_choice_openAI("specific", "get_company_info")))
print("Anthropic (any):", json.dumps(build_tool_choice_anthropic("any")))
print("Gemini (specific):", json.dumps(build_tool_choice_gemini("any", ["get_company_info"])))
```

When shipping a GTM enrichment pipeline (Zone 13 — Production GTM Infrastructure), forced tool choice matters for deterministic workflows. If your Clay table runs a scheduled enrichment that must always call the email-verification tool before finishing, use `tool_choice: "required"` (OpenAI) or `{"type": "any"}` (Anthropic) to prevent the model from skipping the step with a text-only response. The SPF/DKIM/DMARC infrastructure layer that delivers your outbound emails is downstream of this decision — if the enrichment tool is skipped, the email verification never runs, and your deliverability infrastructure has nothing to act on. [CITATION NEEDED — concept: Zone 13 deployment pipeline for Clay tables and n8n workflows]

**Missing tool definitions.** If the model hallucinated a tool name that does not exist in your registry, return an error as the tool result rather than crashing. This is common when models trained on different tool catalogs infer capabilities you have not defined:

```python
def safe_execute(call_id, name, args):
    if name not in TOOLS and name not in ROUTING_TOOLS:
        return {"error": f"Tool '{name}' is not available. Available tools: {list(TOOLS.keys()) + list(ROUTING_TOOLS.keys())}"}
    return execute_tool(name, args)

print(safe_execute("id1", "get_company_info", {"company_name": "Clay", "data_field": "headcount"}))
print(safe_execute("id2", "send_slack_message", {"channel": "#gtm", "text": "hello"}))
```