# Function Calling Deep Dive — OpenAI, Anthropic, Gemini

## Learning Objectives

1. Implement a complete function-calling loop (define → parse → execute → inject → continue) across OpenAI, Anthropic, and Gemini APIs
2. Compare tool definition schemas, response structures, and continuation patterns across all three providers
3. Build a task router that selects and executes tools based on input classification
4. Handle parallel function calls, malformed parameters, and missing tool definitions in production
5. Configure forced tool choice (`auto` / `any` / specific function) and evaluate steering behavior per provider

---

## Beat 1: Hook — The Model Doesn't Execute

The model is a text completer. Function calling is a structured hallucination: the model outputs JSON describing a tool invocation, and your code decides whether to actually run it. This is the loop that turns an LLM into an agent. Without it, you have a chatbot. With it, you have a system that can act.

---

## Beat 2: Concept — The Function Calling Protocol

The protocol has four steps, and every provider implements them with different JSON shapes:

1. **Tool definitions** — You declare available functions with names, descriptions, and parameter schemas (JSON Schema). OpenAI uses `tools[].function`. Anthropic uses `tools[].input_schema`. Gemini uses `functionDeclarations[]`.
2. **Model response** — Instead of (or in addition to) text, the model returns structured tool-call objects. OpenAI: `message.tool_calls[]` with `id`, `function.name`, `function.arguments`. Anthropic: `content[]` blocks of type `tool_use` with `id`, `name`, `input`. Gemini: `parts[]` with `functionCall` containing `name` and `args`.
3. **Local execution** — You parse the call, run your code, and collect the result.
4. **Result injection** — You send the result back as a specific message role. OpenAI: role `tool` with `tool_call_id`. Anthropic: role `user` with `content` block of type `tool_result` and matching `tool_use_id`. Gemini: role `function` with `name` and `response`.

The model then continues — it may call another tool, or it may produce a final text answer.

**Exercise hooks:**
- *Easy:* Print the raw API response from each provider when a tool is called. Identify where the function name, arguments, and call ID live in each JSON structure.
- *Medium:* Write a single `execute_tool(name, args)` dispatcher that all three provider loops share. Confirm it returns identical results regardless of which provider triggered it.

---

## Beat 3: Walkthrough — Same Loop, Three Providers

Build a minimal function-calling agent that uses one tool (`get_company_info`) across all three APIs. Each implementation follows the same algorithm: send messages + tool definitions → check response for tool calls → execute → append result → loop until text-only response.

**OpenAI implementation pattern:** Check `finish_reason == "tool_calls"` in the response. Extract `tool_calls`, iterate, execute each, append `tool` messages with matching `tool_call_id`. Re-send.

**Anthropic implementation pattern:** Check `stop_reason == "tool_use"`. Iterate `content` blocks for `type == "tool_use"`. Execute each, append `tool_result` blocks inside a `user` message with matching `tool_use_id`. Re-send.

**Gemini implementation pattern:** Check response parts for `functionCall`. Execute each, construct `functionResponse` parts. Send back as a `chat.sendMessage` with `functionResponse` parts. Loop.

**Exercise hooks:**
- *Easy:* Run all three implementations with the same input. Print the full request/response cycle for each. Confirm the tool is called once and the final answer incorporates the tool result.
- *Medium:* Modify the tool to return an error (e.g., company not found). Observe how each provider handles the error result in its next turn.
- *Hard:* Implement parallel tool calling — define two tools, craft input that triggers both, and confirm your loop handles multiple tool calls in a single response before re-sending.

---

## Beat 4: Use It — GTM Task Router

**GTM Cluster 09 redirect:** This is the mechanism behind the Agent Stack. A task router is an LLM that classifies incoming work and dispatches it to the right tool. In GTM, that means: "lookup company domain," "enrich contact with email," "score lead against ICP," "draft outreach email." Each of these is a function the model can call.

Build a router with four GTM tools:
- `lookup_domain(company_name) → domain`
- `enrich_contact(email) → {name, title, company}`
- `score_against_icp(company_data) → {score, reasons[]}`
- `draft_email(contact_info, context) → email_text`

The model receives a user prompt like "Research Acme Corp and draft outreach to their VP of Sales." It must call tools in sequence: lookup → score → draft. The function-calling loop handles the orchestration. You do not hardcode the order.

**Exercise hooks:**
- *Easy:* Add the four tool definitions to a single provider's API call. Send the prompt and print which tools the model calls and in what order.
- *Medium:* Implement the full multi-turn loop. The model should call at least three tools before producing a final answer. Print the state after each turn.
- *Hard:* Swap providers mid-loop — start with OpenAI for classification, use a cheaper model for tool execution, and return to the primary model for final synthesis. Document where this breaks.

---

## Beat 5: Ship It — Production Edge Cases

The happy path works. Production doesn't have happy paths.

**Malformed parameters:** The model may return invalid JSON in `arguments` (OpenAI), missing required fields, or wrong types. Wrap every parse in a try/catch. Return an error as the tool result: `"error": "missing required parameter 'company_name'"`. The model will retry.

**Token budget for tool results:** If `lookup_domain` returns 40KB of HTML, you will blow the context window. Truncate, summarize, or paginate before injecting results. This is your responsibility — the API won't do it.

**Hallucinated tools:** The model may call a function you didn't define. This happens more often with complex tool sets. Check the function name against your registry before executing. Return an error if unrecognized.

**Parallel calls and ordering:** OpenAI supports parallel tool calls natively. If tool B depends on tool A's output, you must either (a) prevent parallel execution, or (b) detect the dependency and re-order. Anthropic and Gemini handle this differently — test each.

**Exercise hooks:**
- *Easy:* Send a prompt that triggers a tool call with a missing required parameter. Print the error you return and the model's retry attempt.
- *Medium:* Return a 10,000-character string as a tool result. Measure the token count. Implement truncation to 2,000 characters before injection. Confirm the model still produces a coherent answer.
- *Hard:* Define 10 tools. Send ambiguous input. Log every case where the model calls a tool not in the registry or calls tools in an impossible order. Document failure modes per provider.

---

## Beat 6: Deep Cut — Forced Tool Choice and Provider Steering

Each provider offers a way to force tool selection:

- **OpenAI:** `tool_choice: "auto"` (default), `tool_choice: "required"` (must call at least one), `tool_choice: {"type": "function", "function": {"name": "exact_name"}}` (must call this specific function).
- **Anthropic:** `tool_choice: {"type": "auto"}`, `tool_choice: {"type": "any"}` (must call at least one), `tool_choice: {"type": "tool", "name": "exact_name"}` (must call this one).
- **Gemini:** `tool_config: {function_calling_config: {mode: "AUTO" | "ANY" | "NONE", allowed_function_names: [...]}}`.

Behavior under forced choice is not fully documented. Observable differences:
- When forced to call a specific function, does the model still populate arguments correctly, or does it emit defaults/empty values?
- Does `required`/`any` always call exactly one tool, or can it call multiple?
- What happens when the forced function is irrelevant to the input? Does the model still attempt a coherent response after the irrelevant call?

**Exercise hooks:**
- *Easy:* Set `tool_choice` to force a specific function across all three providers. Send input unrelated to that function. Print the arguments the model generates. Note where it hallucinates plausible values vs. emits empty/defaults.
- *Medium:* Run a benchmark: 20 prompts, each with `auto`, `any`, and forced-specific. Log whether the model calls the "correct" tool (human-labeled) under each mode. Report accuracy per mode per provider.
- *Hard:* Implement a self-correcting loop: if forced tool choice produces an error result, switch back to `auto` and let the model re-plan. Measure whether this recovers or cascades into more errors.

---

## GTM Redirect Summary

This lesson maps directly to **GTM Cluster 09 — Agents, tool use, function calling**. The task router built in Beat 4 is the same pattern used in the **Agent Stack** for workflow automation: the model classifies the task, selects the tool, and your loop executes it. Every n8n/Make node that calls an LLM to decide "what happens next" is implementing this function-calling protocol under the hood. The production edge cases in Beat 5 are what separate a demo from a deployed GTM pipeline.