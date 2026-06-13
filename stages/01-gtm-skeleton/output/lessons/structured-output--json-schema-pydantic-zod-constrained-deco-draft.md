# Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding

## Hook

You've been parsing LLM output with regex and praying. There's a mechanism that eliminates parse failures entirely by constraining what tokens the model can emit during generation.

## Concept

**Constrained decoding** is the core mechanism: at each generation step, the decoder masks logits for tokens that would violate a grammar (JSON Schema compiled to a finite-state automaton). The model physically cannot produce invalid output. JSON Schema is the interchange format. Pydantic (Python) and Zod (TypeScript) are schema-definition ergonomics layers that compile to JSON Schema and validate post-generation. The schema-to-grammar compilation and logit masking happen in the inference engine (Outlines, llama.cpp, vLLM) or the API provider (OpenAI structured output mode).

## Use It

Write Pydantic models for a firmographic extraction task, compile to JSON Schema, and call an LLM endpoint with schema enforcement. Compare raw-output-plus-post-hoc-validation against constrained decoding on the same prompt. GTM redirect: this is the extraction pattern behind Clay's waterfall enrichment — [CITATION NEEDED — concept: Clay's internal structured output mechanism for enrichment columns]. Even without vendor specifics, any GTM system that pulls structured account data from unstructured sources uses this pattern.

## Ship It

Exercise hooks:
- **Easy**: Define a Pydantic model for a company record, serialize to JSON Schema, validate a hardcoded JSON string against it. Print success/failure.
- **Medium**: Call an OpenAI-compatible API with `response_format` set to your schema. Print the parsed output and confirm types match.
- **Hard**: Implement a minimal constrained decoder for a 5-token vocabulary and a schema that allows exactly two valid JSON objects. Show the logit mask at each step.

## Debug It

Constrained decoding failures are qualitatively different from parse failures. Schema too restrictive → model outputs semantically wrong values that happen to be syntactically valid. Nullable fields, unions, and deeply nested arrays cause grammar compilation edge cases. Exercise hook: given a Pydantic model with `Optional` and `Union` fields, show the generated JSON Schema and identify which parts the grammar compiler might struggle with.

## Extend It

Grammar-constrained decoding generalizes beyond JSON — regex grammars, context-free grammars, LARK-based specifications. Zod schemas compile differently than Pydantic schemas for the same logical shape; the differences matter at the inference-engine boundary. Exercise hook: define the same schema in Pydantic and Zod, serialize both to JSON Schema, diff the outputs, and identify which additional properties each adds that a downstream grammar compiler must handle.