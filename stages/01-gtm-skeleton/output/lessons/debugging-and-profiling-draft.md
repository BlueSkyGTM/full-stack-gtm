# Debugging and Profiling

## Hook

When an LLM pipeline fails, the error message is rarely helpful. A 500 from OpenAI tells you nothing about which prompt in your chain caused the failure, how many tokens you burned before it broke, or whether the issue is your logic or the model's output. Debugging AI systems requires tracing non-deterministic flows where the same input can produce different failures on successive runs.

## Concept

Three mechanisms for debugging AI pipelines. First: **structured logging of intermediate state** — capture the exact prompt sent, the raw response received, and the parsed output at each step. Second: **token-level profiling** — measure prompt tokens, completion tokens, and latency per call so you can identify which step in a multi-step chain is burning budget or time. Third: **output validation with retry** — detect malformed LLM outputs (wrong format, missing fields, hallucinated structure) and retry with a corrected prompt that includes the failure signal. Tools that implement these: OpenAI's usage fields in API responses, LangChain callbacks (though the callback mechanism is opaque — you hand over control and trust the framework), and manual wrapper functions that log before/after each call. The skeptical approach: write your own thin wrappers so you control what gets logged.

## Demo

Build a profiling wrapper around an OpenAI chat completion call. The wrapper records: input token count, output token count, latency in milliseconds, the raw prompt, and the raw response. Print a structured log line for each call. Then demonstrate a failing call (e.g., request JSON output but get plaintext back) and show how the logged state reveals the problem.

## Use It

In GTM automation, a Clay waterfall runs contacts through a sequence of enrichment and personalization steps. When a campaign produces low reply rates or high bounce rates, you need the same debugging approach: log which step in the waterfall produced the output, profile token cost per enriched record, and detect where the LLM returned generic copy instead of personalized content. This is the profiling pattern applied to [CITATION NEEDED — concept: Clay waterfall debugging and token cost tracking per enrichment step].

## Ship It

**Easy:** Write a decorator that logs token usage and latency for any function that calls OpenAI.  
**Medium:** Build a multi-step prompt chain (extract → transform → generate) where each step is wrapped with your profiler. Run it on 5 inputs and print a summary table of per-step costs and latencies.  
**Hard:** Implement output validation that detects when an LLM response doesn't match an expected schema. On failure, retry with the original prompt plus the error message appended. Log the retry count and whether the retry succeeded.

## Review

Debugging AI pipelines is not debugging code — it's debugging data flows through a non-deterministic system. The three tools: log intermediate state at every step, profile token usage and latency per call, and validate outputs with retry logic that feeds failures back into the prompt. Without these, you're guessing. With them, you can pinpoint exactly which step in a chain failed, what it cost, and whether retrying fixes it.