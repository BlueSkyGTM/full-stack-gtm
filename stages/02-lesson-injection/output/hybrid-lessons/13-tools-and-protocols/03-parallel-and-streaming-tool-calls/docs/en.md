# Parallel Tool Calls and Streaming with Tools

## Learning Objectives

1. Implement parallel tool calls using the Anthropic Messages API and handle multi-tool-use response blocks in a single turn.
2. Reconstruct complete tool-call arguments from streaming `input_json_delta` chunks, keyed by content block index.
3. Compare wall-clock latency of sequential versus parallel tool execution with measured output.
4. Build a streaming tool-use loop that reconstructs arguments, executes tools concurrently, and returns results within a single streamed exchange.

## The Problem

When an enrichment agent needs to look up a company's funding round, tech stack, and headcount, the naive approach makes three round trips to the model. Each trip looks like this: the model decides to call one tool, your code executes it, you send the result back, the model reads the result and decides to call the next tool. Three independent lookups become serialized across three full model round trips. The wall-clock cost is roughly three times the single-call latency plus three times the executor latency. For a GTM enrichment pipeline processing a hundred accounts, that multiplier turns a two-minute batch into a six-minute batch.

The model already knows it needs all three pieces of data. The constraint is the request-response contract: the non-parallel pattern forces the model to ask one question at a time, wait for the answer, then ask the next. The Anthropic Messages API lifts this constraint by allowing multiple `tool_use` content blocks in a single response. The model emits all three calls at once. Your code executes them concurrently and returns all three results in a single follow-up message. One model round trip instead of three. Executor time becomes the maximum of the three calls, not the sum.

Streaming adds a second dimension. If you stream the model's response — which you want for latency-sensitive interfaces — tool-call arguments arrive as fragments. With a single tool call, reconstruction is straightforward: concatenate the fragments, parse at the end. With parallel tool calls, fragments from different tools interleave on the wire. Block 0 gets a few characters of JSON, then block 1 gets a few characters, then block 0 gets more. You must track which fragment belongs to which block index, accumulate per index, and parse each accumulated string independently. Get the bookkeeping wrong and you merge two tools' arguments into one garbled JSON string.

## The Concept

Parallel tool calling works because the model's output is a list of content blocks, not a single function invocation. When you send `messages.create()` with a `tools` array, the model can respond with `stop_reason: "tool_use"` and a `content` list containing multiple entries of type `tool_use`. Each entry has its own `id`, `name`, and fully parsed `input` dictionary. Your code iterates the list, dispatches each call, collects results, and appends them all as `tool_result` blocks in the next message. The model receives all results at once and produces its final answer. There is no `parallel_tool_calls: true` flag in the Anthropic API — the model decides to parallelize when the calls are independent. (OpenAI's API exposes this as an explicit `parallel_tool_calls` parameter you can disable.)

Streaming changes the parsing surface. With `stream=True`, the response arrives as server-sent events. Each `content_block_start` event signals a new block beginning at a specific index. Each `content_block_delta` event carries an `input_json_delta` with a `partial_json` string — a fragment of the tool's arguments JSON. These fragments are not valid JSON on their own. You concatenate them per index and only parse once `content_block_stop` fires for that index. With multiple parallel calls, the stream interleaves: start block 0, start block 1, delta for block 0, delta for block 1, delta for block 0 again, stop block 1, stop block 0. A dictionary keyed by block index is the standard accumulator.

```mermaid
sequenceDiagram
    participant Code as Your Code
    participant API as Anthropic API
    participant T1 as get_funding
    participant T2 as get_tech_stack
    participant T3 as get_headcount

    Note over Code,API: Non-Streaming Parallel
    Code->>API: create(tools=[funding, tech, head], prompt)
    API-->>Code: stop_reason=tool_use, content=[TU×3]
    par concurrent execution
        Code->>T1: execute
        Code->>T2: execute
        Code->>T3: execute
    end
    T1-->>Code: result
    T2-->>Code: result
    T3-->>Code: result
    Code->>API: create(tool_results=[r1, r2, r3])
    API-->>Code: final text answer

    Note