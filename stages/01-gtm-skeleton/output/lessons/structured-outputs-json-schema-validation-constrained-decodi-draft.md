# Structured Outputs: JSON, Schema Validation, Constrained Decoding

## Hook

You built a pipeline that asks an LLM for JSON. Half the time it works. The other half you get markdown fences, trailing commas, or a chatty preamble about "Here's the JSON you requested." This lesson closes that gap permanently.

## Concept

Three mechanisms, in order of enforcement strength: prompt-only JSON instructions (weakest), JSON mode (token-level constraint that guarantees valid syntax but not valid schema), and structured outputs / constrained decoding (grammar-level enforcement at every generation step). Schema validation with JSON Schema or Pydantic operates after generation as a separate contract layer.

## Demonstration

Working code calling an LLM API three ways—unstructured, JSON mode, and structured output with a supplied schema—with printed output showing exactly what breaks and what doesn't. A second example shows Pydantic validating and rejecting a generated object that was syntactically valid JSON but violated a field constraint.

**Exercise hooks:**
- (Easy) Modify the schema to require an additional field; confirm the model includes it.
- (Medium) Send a malformed response through Pydantic validation and catch the specific error.
- (Hard) Build a schema with nested objects and enums, generate structured output, validate the full tree.

## Use It

GTM cluster: **Enrichment & Waterfall Orchestration (Zone 20)**. Every enrichment step in a Clay waterfall extracts structured data—company name, headcount, funding stage—from unstructured sources. Without structured outputs, the waterfall breaks on parse failures mid-pipeline. Constrained decoding is the mechanism that makes "extract these 5 fields" reliable enough to chain into the next enrichment step without a human reading the output.

## Ship It

Schema versioning when your downstream consumer changes. Fallback strategy: if structured output fails or returns empty fields, what goes in the cell—null, a sentinel, a retry? Rate-limit implications of retries on parse failures. Monitoring: log schema validation rejection rates as a health metric for your enrichment pipeline.

## Evaluate It

Quiz hooks target the mechanism distinction: the difference between JSON mode and structured outputs at the token level, why Pydantic catches errors that JSON mode does not, and which enforcement layer prevents a trailing comma. No trivia—all questions grounded in the demonstrated code behavior.