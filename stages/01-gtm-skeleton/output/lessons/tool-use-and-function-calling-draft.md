# Tool Use and Function Calling

## Learning Objectives

1. Define tool schemas using JSON Schema and pass them to an LLM API call
2. Implement a tool execution loop that parses model responses, executes functions, and feeds results back
3. Compare `tool_choice` modes (`auto`, `required`, `none`, specific function) and predict model behavior for each
4. Handle tool execution errors without breaking the conversation loop
5. Calculate the token cost of tool definitions and trim schemas for production use

---

## Beat 1: Hook

You have a model that can reason about what should happen, but cannot make it happen. Tool use is the bridge: the model emits structured instructions, your code runs them, and the result goes back into context. This is the loop that turns a chatbot into an agent.

---

## Beat 2: Concept

Function calling is a constrained output pattern. Instead of generating freeform text, the model generates a JSON object containing a function name and arguments. Your code parses that object, executes the function, and appends the result to the conversation. The model then continues with that new information. The model does not run the function. It predicts which function should run and what arguments to pass.

Key terms:
- **Tool definition**: A JSON Schema describing a function's name, description, and parameters. Sent with the API request.
- **Tool call**: The model's structured output requesting a function execution.
- **Tool result**: Your code's output, passed back as a message with `role: "tool"`.

---

## Beat 3: Mechanism

**How the model decides to call a tool:**
The tool definitions are injected into the system context as structured metadata. During inference, the model can emit special tokens that indicate "I want to call a function" followed by the function name and arguments. This is still next-token prediction, but over a constrained grammar that produces valid JSON instead of prose.

**The loop:**
1. You send messages + tool definitions to the API
2. The model responds with either text or a `tool_calls` array
3. If `tool_calls`: you execute each function, collect results
4. You append tool results as `role: "tool"` messages
5. You call the API again with the updated conversation
6. Repeat until the model responds with text (no more tool calls)

**`tool_choice` modes:**
- `auto`: Model decides whether to call a tool or respond with text
- `required`: Model MUST call at least one tool (never responds with text)
- `none`: Model is forbidden from calling tools (ignores tool definitions)
- `{"type": "function", "function": {"name": "..."}}`: Model must call this specific function

**Token cost reality:**
Every tool definition is included in every API request. A schema with verbose descriptions and nested objects can add 500-2000 tokens per request. In a 10-turn loop, you pay that cost 10 times. Trim descriptions and remove unused parameters.

---

## Beat 4: Code

**Example 1: Single tool call**
Define one tool (`get_weather`), send it with a user message, parse the model's tool call, execute a mock function, return the result, and print the model's final response.

**Example 2: Multi-turn tool loop**
Define three tools (`lookup_company`, `get_funding`, `find_contacts`). The user asks a research question. The loop runs until the model stops calling tools or hits a maximum iteration count. Each iteration prints the tool call and result.

**Example 3: Error handling**
A tool call with invalid arguments triggers a graceful error response fed back as a tool result with `is_error: true`. The model recovers and retries with corrected arguments.

**Exercise hooks:**
- Easy: Add a new tool to the three-tool example and test it with a prompt that triggers it
- Medium: Implement a max-iteration circuit breaker that returns a summary instead of looping forever
- Hard: Build a tool-use agent where one tool's output becomes the input to the next tool's arguments, with no hardcoding of the sequence

---

## Beat 5: Use It

**GTM redirect: Enrichment workflows (Zone 2 — Enrich)**

The Clay waterfall is a tool-use pattern. Each step in the waterfall is a tool definition: "check provider A for email, if null check provider B, if null check provider C." The orchestration logic that decides which tool to call next based on null results is what makes it a waterfall rather than a parallel fan-out.

When you build a research agent that looks up company firmographics, pulls funding data, and finds decision-maker contacts, you are implementing a multi-tool loop. The model decides the sequence. Your code executes each step and returns results.

[CITATION NEEDED — concept: Clay waterfall implementation pattern and tool-call sequencing]

**Exercise hooks:**
- Easy: Define tool schemas for three enrichment sources (company lookup, funding check, contact finder) and test with mock data
- Medium: Implement fallback logic where a null result from one tool triggers a call to an alternative tool
- Hard: Build a full enrichment agent that takes a domain, enriches company data with fallback sources, and outputs a structured record

---

## Beat 6: Ship It

**Production considerations:**

1. **Timeouts**: Tool execution can hang. Wrap every function call in a timeout. Return an error tool result if the function exceeds the limit.

2. **Schema validation**: The model can generate malformed arguments. Validate against your JSON Schema before executing. Reject early, feed the error back.

3. **Idempotency**: If your loop retries on network failure, the same tool call may execute twice. Functions that write data (create a CRM record, send an email) must be idempotent.

4. **Cost monitoring**: Log the token count of your tool definitions. If your schemas grow past 1000 tokens, audit for unnecessary descriptions. Each reduction saves tokens on every turn.

5. **Rate limiting**: A multi-tool loop can make 5-10 API calls per user request. Implement request-level rate limiting and backpressure.

**Exercise hooks:**
- Easy: Add a timeout wrapper to every tool execution and return an error result on timeout
- Medium: Add schema validation that catches malformed arguments and returns a descriptive error to the model
- Hard: Instrument a multi-tool loop with token counting, latency tracking, and cost estimation per turn

---

## GTM Redirect Rules

This lesson maps to **Zone 2 (Enrich)** in the GTM topic map. The specific redirect is: multi-source enrichment workflows where each data provider is a tool definition, and fallback logic is tool-call sequencing. When the lesson references "enrichment agent" or "waterfall," it is referencing this pattern, not a generic "AI is useful for GTM" claim.

If the AI concept does not cleanly map to a GTM application in a given section, the redirect defaults to: "foundational for building agents that execute GTM workflows" without fabricating a specific use case.