# The Tool Interface — Why Agents Need Structured I/O

## Hook
An agent that receives a string when it expects a list, or a float when it expects a boolean, halts or hallucinates. This beat opens with a minimal agent that breaks on unstructured tool output, then shows the same agent succeeding with a typed schema. The contrast establishes the stakes: without a contract, tool-calling is unreliable.

## Concept
Define the tool interface as a bidirectional contract: input schema (what the agent must provide) and output schema (what the tool must return. Introduce JSON Schema as the lingua franca for describing shape, type, and constraints. Explain that LLM function-calling APIs (OpenAI, Anthropic) are just JSON Schema consumers. The agent reasons over the schema to decide *what* to call; the tool validates against the schema to decide *whether* to accept.

## Mechanism
Walk through the three-stage lifecycle: schema definition → agent serialization → tool validation. Show how an LLM populates a function's `parameters` object from the schema, how the receiving function parses and validates, and what happens when validation fails. Demonstrate with a working Python example that defines a schema, simulates an LLM call populating it, and validates the result with `jsonschema`. Observable output: printed schema, populated payload, validation result (pass/fail with error messages on failure).

*Exercise hooks:*
- Easy: Modify one field type in the schema and observe the validation error.
- Medium: Add a required enum constraint to a field; break it, then fix it.
- Hard: Build a two-tool agent where Tool A's output schema is fed as input to Tool B, and prove the chain fails if schemas mismatch.

## Use It
GTM redirect: **Zone 2 — Data Enrichment, specifically the Clay waterfall pattern.** Each enrichment step in a waterfall is a tool with a structured interface: it expects a domain (string), returns a set of fields (typed). If Clearbit returns `{"employees": "500-1000"}` (string range) and the next tool expects `{"employees": int}`, the waterfall stalls. Show how Clay's architecture treats each enrichment provider as a tool with a declared output schema, and why mapping between providers is essentially schema negotiation.

## Ship It
Build a minimal tool registry in Python: a decorator that attaches a JSON Schema to any function, a validator that checks inputs before execution, and a dispatcher that an LLM can call. The registry prints a manifest of available tools (names + schemas), simulates an LLM selecting one, validates the call, and prints the result. This is the skeleton every agent framework (LangChain, CrewAI, AutoGen) implements internally.

*Exercise hooks:*
- Easy: Add a second tool to the registry and print the updated manifest.
- Medium: Implement error handling that returns a structured error object (not an exception) when validation fails, so the agent can retry.
- Hard: Add a `returns` schema to each tool and validate the tool's output before returning it to the agent. Break a tool by returning the wrong type; prove the output validation catches it.

## Evaluate
Three to five questions grounded in observable behavior from the Mechanism and Ship It code. Questions should test: what happens when schema validation fails, why JSON Schema (not Python type hints) is the contract layer, how an LLM uses the schema during function calling, and what breaks in a multi-tool chain when schemas don't align. No trivia — every question references code output or a failure mode demonstrated in the lesson.