# Tool Registry with Schema Validation

---

## Beat 1: Hook — When Tools Bite Back

You wired a web search tool into your agent. It worked in testing. In production, someone passed a negative `limit` parameter and the API returned 500 results instead of 10—then billed you for all of them. Schema validation is the contract layer that prevents this. Every tool you register needs one: a declarative spec that says what inputs are legal, what types they carry, and what the output shape looks like. Without it, you're shipping an agent with no guardrails on its hands.

---

## Beat 2: Concept — The Registry-and-Validate Pattern

A **tool registry** is a lookup table: tool name → callable function + input schema + output schema. When an agent decides to call a tool, the registry intercepts the call, validates the proposed arguments against the input schema, executes only if valid, and validates the return against the output schema before passing it back. The schemas are typically JSON Schema objects. The registry pattern separates *declaration* (what the tool expects) from *execution* (what the tool does). This is the same pattern OpenAI uses internally when you define `functions` in the API—each function has a `parameters` field that is a JSON Schema. The mechanism is: schema-defined contract → pre-execution validation → execution → post-execution validation → result or error.

---

## Beat 3: Demo — Build a Registry from Scratch

Working Python implementation of a `ToolRegistry` class that:
- Accepts tool registrations with callable + input_schema (JSON Schema) + output_schema
- Validates inputs before execution using `jsonschema`
- Validates outputs after execution
- Returns structured errors on validation failure
- Includes 3 registered tools (web_search, get_contact, send_email) with realistic schemas
- Runs 5 test calls (3 valid, 2 invalid) and prints all results to confirm pass/fail

**Observable output:** printed validation results for each call showing which passed, which failed, and why.

---

## Beat 4: Use It — Schema-Gated Enrichment Tools

GTM cluster: **Zone 2 — Enrichment**, specifically the enrichment waterfall pattern in Clay.

When Clay runs a waterfall over multiple enrichment providers, each provider is effectively a tool in a registry. The input schema enforces that you don't send a `domain` field to a provider that requires `linkedin_url`. The output schema enforces that downstream routing logic receives the shape it expects. Building your own tool registry with schema validation is the same mechanism Clay uses to gate which enrichment columns get populated and which get skipped. [CITATION NEEDED — concept: Clay internal enrichment validation schema structure]

**Exercise hooks:**
- *Easy:* Register a `lookup_company` tool with input schema requiring `domain` (string, format: uri) and output schema returning `{name, employee_count, industry}`. Call it with valid and invalid inputs.
- *Medium:* Build a mini-enrichment waterfall: register 3 tools (Clearbit-style, Hunter-style, Apollo-style) that accept the same input schema but return different output shapes. The registry tries each in sequence until one returns valid output.
- *Hard:* Add `anyOf` and conditional schema fields. Register a tool that accepts EITHER `email` OR `domain` but requires `company_name` if `domain` is provided. Validate that the conditional logic works by printing the validation path taken.

---

## Beat 5: Debug It — When Schemas Lie

Common failure modes:
- **Schema too permissive:** `{}` accepts everything. Tool runs, hits a runtime error on a missing field. The schema said it was fine. Lesson: always define `required` fields and `additionalProperties: false`.
- **Schema too strict:** `pattern` regex rejects valid inputs. A company name with an ampersand (`&`) fails a `^[a-zA-Z0-9 ]+$` pattern. Lesson: test schemas against real data, not clean data.
- **Output schema drift:** You defined the output schema when you registered the tool. The upstream API changed its response shape. Now every call fails post-execution validation. Lesson: log the actual output on validation failure so you can update the schema.
- **Numeric edge cases:** `minimum: 0` allows `0` but not `-1`. Your agent passes `limit: 0` meaning "no limit." The API returns everything. Lesson: schemas validate structure, not intent.

**Exercise hooks:**
- *Easy:* Given a broken schema (missing `required`), show how a tool call with an empty object passes validation but crashes at runtime. Fix the schema.
- *Medium:* A tool's upstream API changed from `{email, name}` to `{email, full_name}`. The output validation fails silently. Add logging to surface the mismatch between expected and actual output shape.
- *Hard:* Build a schema migration helper: given an old schema and a sample of real outputs, detect which fields have drifted and propose an updated schema.

---

## Beat 6: Ship It — Production Registry with Error Recovery

Production requirements for a tool registry:
- **Graceful degradation:** If validation fails, return a structured error the agent can reason about, not a stack trace.
- **Schema versioning:** Tools change. Schemas change. Tag schemas with versions so you can migrate without breaking existing agent prompts.
- **Rate-limit awareness:** The registry should track per-tool call counts and reject calls that exceed limits before execution, not after.
- **Observability:** Every validation pass/fail and every execution result logged as a structured event.

Working implementation: extend the demo registry with `ToolResult` dataclass (success/failure + data + validation_errors), versioned schemas, and a call logger that prints structured JSON lines.

**Exercise hooks:**
- *Easy:* Wrap all tool calls in try/except. On validation error, return a `ToolResult` with `success=False` and a human-readable error message instead of raising.
- *Medium:* Add schema versioning. Register `lookup_company` as v1 and v2 with different output shapes. The registry routes to the latest version by default but accepts a `version` parameter. Print which version was used.
- *Hard:* Build the full production registry with: versioned schemas, per-tool rate limiting (max 5 calls per tool per session), structured JSON logging of every validation event, and a summary function that prints call statistics at the end.

---

**GTM Redirect Summary:**
This lesson is **foundational for Zone 2 (Enrichment)**. The registry-and-validate pattern is the same mechanism that governs enrichment waterfalls in Clay: schema-gated inputs control which providers receive which data, and schema-validated outputs control which downstream columns get populated. [CITATION NEEDED — concept: Clay waterfall validation architecture]