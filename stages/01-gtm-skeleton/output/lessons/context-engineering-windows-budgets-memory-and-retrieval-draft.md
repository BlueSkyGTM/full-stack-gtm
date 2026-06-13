# Context Engineering: Windows, Budgets, Memory, and Retrieval

## Hook

Every LLM call has a hard ceiling: the context window. Token budget determines what fits. Retrieval determines what enters the window. This lesson breaks the four constraints that govern every prompt you ship.

## Concept

**Context Window Mechanics.** The context window is the maximum number of tokens a model can process in a single call. It is not "memory" — it is a fixed-size buffer. Explain the distinction between input tokens (prompt + retrieved context + conversation history) and output tokens (model response). Introduce the budget equation: `input_tokens + output_tokens ≤ context_window`. Demonstrate what happens at the boundary — truncation, error, or silent failure depending on provider.

**Token Budgeting.** Token count ≠ character count. Describe the BPE (Byte Pair Encoding) mechanism at a high level: common subwords get single tokens, rare words fragment. Show that "I love cats" and "antidisestablishmentarianism" occupy different token counts despite similar character counts. Introduce `tiktoken` as the OpenRAIL tokenizer that estimates token counts before API calls. Budget discipline: reserve tokens for output, allocate remaining budget across system prompt, retrieved context, and conversation history.

**Memory Architectures.** Define three patterns: (1) **Full history** — stuff every prior turn into the prompt, scales linearly, hits the ceiling fast. (2) **Sliding window** — keep the last N turns, oldest turns fall off. (3) **Summarization** — compress older turns into a summary, append recent turns verbatim. Each trades off recall fidelity for token efficiency.

**Retrieval-Augmented Context.** When the needed information exceeds the window or is not in the conversation, retrieval fills the gap. Describe the RAG loop: query → embedding similarity search → top-K results → inject into prompt → generate. The retrieval step is a budget decision: every chunk you retrieve consumes tokens that could have been used for instructions or output.

## Demo

Write a Python script that takes a sample conversation, tokenizes it with `tiktoken`, and prints a budget report showing how many tokens each segment (system prompt, history turns, retrieved context, output reservation) consumes relative to a stated window limit. The script prints a warning when the budget exceeds the window. [CITATION NEEDED — concept: token budget accounting across prompt segments in production systems]

Second script demonstrates the three memory strategies on the same conversation: full history (shows total tokens), sliding window last 3 turns (shows reduced tokens), and summarized history (shows compressed tokens with a mock summary). Prints a comparison table.

## Use It

**GTM Redirect: Zone 1 — Account Research Enrichment.** When enriching accounts via Clay or similar tools, each LLM call to summarize research, score intent, or draft personalization must operate within the context window. If you pull 10 SEC filings into a prompt, you will exceed most model windows. Retrieval (embeddings + search) selects the relevant chunks; budget discipline ensures room for the output. The Clay waterfall pattern — sequential enrichment where each step adds a field — is a form of managed context: each step produces structured output that becomes input to the next, and token budgets determine how much accumulated context survives the chain.

**Exercise (Easy):** Given a prompt template and a token limit, calculate whether adding one more retrieved chunk will exceed the budget. Print the decision.

**Exercise (Medium):** Build a sliding-window conversation buffer that accepts new turns and automatically prunes the oldest when the estimated token count exceeds a threshold. Print the buffer state after each turn.

**Exercise (Hard):** Implement a retrieval selector that takes a list of document chunks with token counts and a budget, then selects chunks in relevance order until the budget is exhausted. Print which chunks were selected and the total tokens used.

## Ship It

Build a token budget auditor for any prompt workflow. Input: a JSON array of prompt segments (system, context, history, instruction), each with content and a priority level. Output: a report showing total tokens, per-segment tokens, and a recommended trim order (lowest priority first) if over budget. The script prints a "pass" or "trim needed" verdict with specifics.

**GTM Redirect:** This auditor pattern applies directly to Clay enrichment workflows where multiple data sources (LinkedIn, company website, news) are concatenated into a single LLM call. Budget overruns cause silent truncation or API errors. The auditor catches this before the call.

## Evaluate

**Discussion prompt:** When a retrieval system returns 15 chunks but the budget only allows 5, what is the failure mode for the end user? How would you detect this in production logs without reading every generated response?

**Exercise hook (not full text):** Write a test that asserts a prompt builder never returns a token count exceeding the configured window minus the output reservation. The test takes a list of mock retrieval results of varying sizes and confirms the builder handles overflow correctly.