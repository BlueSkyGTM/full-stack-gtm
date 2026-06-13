# Function Calling & Tool Use

## Learning Objectives

1. Implement a function-calling loop that parses LLM structured outputs into executable tool invocations.
2. Compare the OpenAI function-calling protocol against Anthropic's tool-use schema format.
3. Build a multi-tool agent that chains tool calls based on intermediate results.
4. Detect and handle tool-call failures with retry and fallback logic.
5. Wire a tool-calling agent into a GTM workflow node (enrichment, routing, outreach trigger).

---

## Beat 1: Hook It

The gap between "LLM generates text" and "LLM takes action." Why raw text output breaks when you need to call an API, query a database, or trigger a workflow. The real mechanism: structured output as a control plane, not a chat interface.

---

## Beat 2: Ground It

The call-decide-execute loop. How tool schemas work (JSON Schema as contract between model and code). The lifecycle: system prompt declares available tools → model returns a tool-use message → your code executes the function → result feeds back as a tool-result message. Differences between OpenAI's `function_call` / `tool_calls` and Anthropic's `tool_use` content blocks. Why the model never runs the tool itself — your code is the runtime.

---

## Beat 3: Build It

Working implementation of a minimal tool-calling agent. Define two tools (e.g., `lookup_company` and `score_lead`), register their schemas, and run the loop: send messages with tool definitions, parse the model's tool-use response, execute locally, append the result, and re-prompt. Full runnable code. Observable output: print each step of the loop so the practitioner can trace the decision chain.

**Exercise hooks:**
- **Easy:** Add a third tool to the existing agent and observe how the model selects it.
- **Medium:** Modify the loop to handle cases where the model requests multiple tool calls in a single turn.
- **Hard:** Implement a max-turns guard and a "no tool called" exit condition, then test edge cases where the model should stop calling tools.

---

## Beat 4: Use It

**GTM Cluster 09 redirect:** This is the mechanism behind every enrichment waterfall and task-routing agent. In a Clay waterfall, each step is a tool call — the router decides what fires next. In n8n/Make, a webhook receives a lead, your agent calls `enrich_company`, then `check_intent`, then `draft_outreach`, each mapped to a node. The algorithm you built in Beat 3 is the same loop that powers Cold Calling Infrastructure (Zone 2.2): fetch lead data → score → queue dial. Map each tool to a GTM action. [CITATION NEEDED — concept: Clay waterfall tool-call internals]

**Exercise hooks:**
- **Easy:** Replace the demo tools with two GTM-relevant tools: `enrich_domain` (mock) and `generate_personalized_line` (mock).
- **Medium:** Chain three mock GTM tools so the output of `enrich_domain` feeds into `score_icp_fit`, which feeds into `draft_email`.
- **Hard:** Build a task router that inspects the lead's industry field and conditionally skips the enrichment step if the data is already present.

---

## Beat 5: Ship It

Production concerns. Token cost of tool definitions in every message. Latency of multi-turn tool loops. Rate-limit handling when a tool calls an external API. Schema validation — what happens when the model returns malformed JSON (it will). Retry strategies: re-prompt with the error message vs. fallback to a default. Logging every tool call for debugging and audit. Security: tool schemas as an attack surface (prompt injection via tool descriptions).

**Exercise hooks:**
- **Easy:** Add error handling that catches a tool execution failure and sends the error back to the model as a tool-result.
- **Medium:** Implement a retry wrapper that re-executes a failed tool up to N times before surfacing the error.
- **Hard:** Build a logger that records every tool call (name, arguments, result, latency) to a JSONL file, then inspect the trace for a 5-turn agent run.

---

## Beat 6: Push It

Parallel tool calls — when the model requests multiple independent tools in one turn and you execute them concurrently. Forced tool choice (`tool_choice: required` / `any` / specific tool name) and when to use each. Agents that write their own tool schemas at runtime. The boundary between a tool-calling loop and a full agent framework (ReAct, Plan-and-Execute). Where this goes next: MCP (Model Context Protocol) as a standardized tool layer.

**Exercise hooks:**
- **Easy:** Configure `tool_choice` to force the model to always call a specific tool first.
- **Medium:** Implement parallel execution of multiple tool calls returned in a single model response, then compare latency against sequential execution.
- **Hard:** Build a two-agent system where Agent A decides which tools to call, and Agent B validates the tool outputs before they're returned to Agent A.