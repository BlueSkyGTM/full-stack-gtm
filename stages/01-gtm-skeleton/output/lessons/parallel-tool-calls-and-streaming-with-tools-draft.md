# Parallel Tool Calls and Streaming with Tools

## Learning Objectives

1. Implement parallel tool calls using the Anthropic Messages API and handle the resulting multi-tool-use response blocks.
2. Reconstruct complete tool-call arguments from streaming delta chunks.
3. Compare latency of sequential vs. parallel tool invocations with measured output.
4. Build a streaming tool-use loop that executes tool calls and returns results within a single streamed exchange.

---

## Beat 1: Hook

**The problem: sequential tool calls are slow.** When an agent needs to look up a company's funding round *and* their tech stack *and* the hiring plan, doing those three calls one after another triples your wall-clock time. The Anthropic API can return multiple `tool_use` blocks in a single response — meaning the model decides to call several tools at once, and you execute them concurrently. This lesson covers both the parallel-call mechanism and the added complexity of streaming when tool calls are involved.

---

## Beat 2: Concept

**Mechanism: how parallel tool calls work.** When you send a message with tools defined, the model can return multiple `tool_use` content blocks in a single `stop_reason: "tool_use"` response. Each block has a unique `id`, a `name`, and parsed `input` JSON. Your code iterates over all of them, executes each tool, then appends *all* `tool_result` blocks to the conversation before calling the model again.

**Mechanism: how streaming changes tool parsing.** In streaming mode (`stream: true`), tool-call arguments arrive as `input_json_delta` chunks — partial JSON strings you must concatenate. The `content_block_start`, `content_block_delta`, and `content_block_stop` events bracket each tool call. With multiple parallel calls, you'll see these events interleaved: start block 0, start block 1, delta block 0, delta block 1, stop block 0, stop block 1. You accumulate text per block index.

**Key difference from single tool use:** non-streaming gives you fully parsed `input` as a dict. Streaming gives you string fragments you must join and `json.loads` yourself.

---

## Beat 3: Code

**Three runnable examples:**

1. **Parallel tool calls (non-streaming).** Define three tools (`get_funding`, `get_tech_stack`, `get_headcount`). Send a prompt that triggers all three. Iterate the `tool_use` blocks, execute mock functions, return all `tool_result` blocks. Print each tool's latency and the total wall-clock time.

2. **Streaming tool calls.** Same setup, but with `stream=True`. Accumulate `input_json_delta` fragments keyed by content block index. After stream ends, parse each accumulated string into JSON. Print the reconstructed arguments and confirm they match expected values.

3. **Latency comparison.** Run the same three-tool prompt both ways (sequential — three separate API calls — vs. parallel — one call). Print a side-by-side latency table.

**Exercise hooks:**
- *Easy:* Add a fourth tool and confirm it appears in the parallel output.
- *Medium:* Modify the streaming accumulator to handle a tool that returns mid-stream errors (simulate a `partial_json` parse failure).
- *Hard:* Build a generic `stream_and_execute` function that handles any number of parallel tool calls, executes them concurrently with `asyncio.gather`, and feeds results back in a single follow-up message.

---

## Beat 4: Use It

**GTM redirect: parallel enrichment.** In Clay and similar platforms, enriching a lead often means hitting multiple data providers — Clearbit for firmographics, Hunter for email, LinkedIn for title. Each is an independent tool call. Doing them in parallel cuts enrichment time from seconds-per-field to one round trip. This is the mechanism behind Clay's "enrichment waterfall" when multiple providers are queried simultaneously rather than sequentially as fallbacks.

[CITATION NEEDED — concept: Clay parallel enrichment implementation, whether Clay fires multiple enrichment providers concurrently or in strict waterfall sequence]

**Exercise hook:** *Medium:* Define three enrichment tools (firmographics, email finder, social profiles). Send a prompt with a company domain. Print the parallel results and total latency. Compare to sequential latency.

---

## Beat 5: Ship It

**Build a streaming parallel enrichment agent.** Combine everything: streaming output for the final text response, parallel tool calls for data lookups, concurrent execution of those tools. The agent should: (1) stream the model's reasoning text to the terminal, (2) detect parallel tool calls mid-stream, (3) execute all tools concurrently, (4) feed results back, (5) stream the final summary. Print timestamps at each stage.

**Exercise hooks:**
- *Easy:* Ship the agent with two tools and confirm streaming output appears token-by-token.
- *Hard:* Add error handling — if one tool fails (timeout), the agent still proceeds with the other results and streams a note about the failure.

---

## Beat 6: Evaluate

**Questions grounded in this lesson's mechanisms:**

1. In a streaming response with two parallel tool calls, what events delimit each tool's arguments? Describe the interleaving pattern.
2. Why must you accumulate `input_json_delta` strings rather than accessing a parsed `input` dict during streaming?
3. When you receive multiple `tool_use` blocks in one response, what must you do before your next API call? (Specific: the shape of the messages array.)
4. What happens to wall-clock time if you execute three independent tool calls sequentially vs. receiving them as parallel blocks and running them with `asyncio.gather`? Quantify with the latency comparison output from Beat 3.

---

## GTM Redirect Rules (repeated for "Use It" and "Ship It")

- **Cluster:** Zone 1–2 enrichment (data providers, firmographic + contact enrichment)
- **Specific redirect:** parallel tool calls are the mechanism that lets a Clay-style enrichment waterfall query multiple providers in one round trip, rather than one provider at a time
- **Foundational note:** if the GTM platform does not expose parallel execution to the user, this concept is foundational for understanding *why* enrichment latency varies and *how* API-level concurrency works under the hood