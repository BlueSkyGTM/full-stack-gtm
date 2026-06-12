# Phase 13 — Tools and Protocols (quiz factory)

## Focus

The tooling and protocol layer that connects LLMs to the world: function calling / tool use, the Model Context Protocol (MCP), JSON Schema and structured outputs, retrieval APIs, code execution sandboxes, and the agent-tool loop.

## Scrape hints

- `docs/en.md`: MCP spec sections (transport, JSON-RPC, resource/prompt/tool descriptors), function-calling schema shapes, sandbox isolation guarantees
- `code/main.py`: often demonstrates an MCP server stub, a tool dispatcher, or a function-call validation loop
- Vocabulary: `glossary/terms.md` for JSON-RPC, stdio transport, tool descriptor, structured output, MCP resource, prompt template

## Style anchor

- No gold quiz in this phase yet — use `phases/07-transformers-deep-dive/15-attention-variants/quiz.json` for structural reference
- pre = why unstructured tool calls fail (parsing, safety), check = protocol mechanism, post = code demo (the lesson's MCP server or dispatcher)

## Common distractor patterns

- Confuse MCP resource (data source exposed to model) with MCP tool (function the model can call)
- Confuse JSON-RPC request/response with REST HTTP semantics
- Confuse stdio transport (process pipes, no network) with SSE / WebSocket transports
- Mix function-calling (model emits a JSON block) with structured output (constrained decoding)
- Conflate tool descriptor (schema defining callable parameters) with system prompt injection

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
