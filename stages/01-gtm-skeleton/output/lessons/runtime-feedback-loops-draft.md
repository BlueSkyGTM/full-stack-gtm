# Lesson: Runtime Feedback Loops

## Hook

Runtime feedback loops are the mechanism that separates a fire-and-forget script from a system that corrects itself. When an LLM returns malformed JSON, a classification produces an invalid label, or a chain produces a confidence score below threshold, the system needs a path other than "fail silently." This lesson covers the pattern of detecting bad outputs at runtime and routing them back through the pipeline with corrective context — without human intervention.

## Concept

A runtime feedback loop has three components: a **validator** that checks output against a schema or constraint, a **feedback injector** that appends error context to the next input, and a **termination condition** that prevents infinite retry cycles. The mechanism is a while-loop with state mutation — each iteration accumulates error history so the model sees what went wrong. The critical design decision is the failure mode: the loop must eventually give up and return a structured error rather than retry indefinitely. This is not fine-tuning. The model's weights do not change. The loop changes the *input context* based on the *output failure*.

## Code

Build a synchronous feedback loop that asks an LLM to extract structured data from a raw company description. The validator checks for required fields (company_name, industry, employee_range). If validation fails, the error message is injected into the next prompt as context. The loop terminates after three attempts and returns the last error. All output is printed to confirm the loop executed and either succeeded or exhausted retries. Two code examples: one using a local deterministic function to simulate the LLM (runs without API key), one showing the actual API call structure for Claude or OpenAI with a clear marker that it requires a key.

**Exercise hooks:**
- **Easy:** Modify the validator to add a new required field and observe the retry behavior.
- **Medium:** Replace the static max-retries with a confidence-threshold termination — simulate confidence scores and stop only when above 0.8.
- **Hard:** Implement a branching feedback loop where validation failure routes to a *different* prompt template (e.g., a "simpler extraction" prompt) rather than retrying the same prompt with error context.

## Use It

This is the mechanism behind the enrichment waterfall in [CITATION NEEDED — concept: enrichment waterfall retry pattern, likely Zone 2 enrichment pipeline]. When a Clay waterfall enriches a company profile and the first data provider returns incomplete fields, the system does not stop — it routes to the next provider with context about what is missing. The feedback loop pattern here is identical: validate output, inject failure context, retry with adjusted input. The same pattern applies when enriching leads: if firmographic extraction returns `"industry": null`, the loop catches it, notes the missing field, and re-prompts with narrower instructions. The GTM-specific tuning is in the *validator*: your validation rules encode your data contract for downstream CRM ingestion — the feedback loop enforces that contract at runtime.

## Ship It

In production, feedback loops require three safeguards: a hard retry ceiling (never exceed it, even if the error is "almost fixed"), a latency budget (each retry adds seconds; set a wall-clock timeout), and a logging contract (every iteration logs the input, output, validation result, and retry count to structured storage). Without these, a feedback loop becomes a latency bomb. Monitor retry distributions: if 80% of requests succeed on attempt 1, your system is healthy. If the median is attempt 2 or 3, the prompt or model needs revision — the feedback loop is compensating for a weak initial design. Alert on any request that hits the retry ceiling; that is a data quality gap.

## Troubleshoot

- **Loop retries infinitely on edge cases:** The termination condition is missing or the error context is not actually changing the model's behavior. Print the full prompt at each iteration to confirm the feedback injection is present.
- **Model repeats the same mistake across retries:** The error context is too vague. Instead of `"output was invalid"`, inject the specific schema violation: `"company_name is required but you returned null. The raw input was: ..."`.
- **Latency spikes in production:** The retry ceiling is too high or the model is slow. Reduce max retries to 2 and add a wall-clock timeout of 10 seconds per iteration.
- **Validation passes but downstream consumer rejects the data:** The validator's schema does not match the consumer's contract. Align them — the validator *is* the contract.

---

**Learning Objectives (draft):**
1. Implement a while-loop feedback mechanism that validates LLM output and re-prompts with injected error context.
2. Configure a termination condition that prevents infinite retry cycles.
3. Diagnose when a feedback loop is compensating for a weak prompt versus handling genuine edge cases.
4. Apply the feedback-loop pattern to an enrichment pipeline where validation failures trigger re-extraction with narrowed instructions.