# Tool Schema Design — Naming, Descriptions, Parameter Constraints

## Hook It
Show two tool schemas hitting the same LLM: one with vague naming and missing constraints, one with precise naming, tight descriptions, and constrained parameters. The first produces a failed API call; the second succeeds. The entire difference is the schema.

## Ground It
Define the tool schema as the contract between an LLM and an external action: the function name, the natural-language description, and the JSON Schema that constrains each parameter's type, allowed values, and optionality. This is the surface area the model can "see" when deciding whether and how to call a tool.

## Explain It
Walk through three dimensions. **Naming**: verb-noun patterns (`lookup_company`, not `company`), prefix conventions for avoiding collision. **Descriptions**: what the tool does, when to use it, what it returns — the model reads this to disambiguate between tools. **Parameter constraints**: `type`, `enum`, `pattern`, `minimum`/`maximum`, `required` — every constraint you add shrinks the space of malformed calls. Mechanism: the LLM does not "run" the tool; it emits JSON conforming to the schema, and stricter schemas produce higher-fidelity outputs because they reduce ambiguity at inference time. JSON Schema is the standard; OpenAI, Anthropic, and Mistral all accept variants of it.

## Use It
Build a tool schema for a GTM enrichment function — e.g., `enrich_contact` that takes a LinkedIn URL or email and returns firmographic data. This maps to the enrichment waterfall pattern in Clay [CITATION NEEDED — concept: Clay tool schema format for HTTP API integrations], where each step in the waterfall is a tool the agent may call. A well-designed schema is what prevents the model from passing a company name into a field that expects a domain.

## Ship It
Define and validate three tool schemas against a local JSON Schema validator, then print the validation results. Each schema demonstrates a different failure mode: missing required field, value outside enum, wrong type. Output confirms which calls would have been rejected before ever reaching an API.

## Drill It
- **Easy**: Given a poorly-named tool schema, rename the function and parameters to follow verb-noun convention; print before and after.
- **Medium**: Write a parameter constraint that rejects free-text input and forces enum selection for a `lead_status` field; validate sample inputs.
- **Hard**: Design a complete three-tool schema set for a GTM workflow (lookup, enrich, write-back) where two tools share a parameter name with different allowed values; demonstrate that the descriptions disambiguate correctly when submitted to an LLM function-calling endpoint.

---

**Learning Objectives (draft):**
1. Write tool function names using verb-noun patterns that disambiguate intent.
2. Author tool descriptions that specify trigger conditions, expected inputs, and return shape.
3. Define JSON Schema parameter constraints (`type`, `enum`, `required`, `pattern`) that reject invalid inputs at the schema layer.
4. Diagnose malformed LLM tool calls traceable to missing or loose schema constraints.
5. Build a multi-tool schema set where shared parameter names are disambiguated by description and constraint differences.