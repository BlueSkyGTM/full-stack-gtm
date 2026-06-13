# Task Spec Format

## Hook It
A vague prompt produces garbage output at scale. A task spec constrains the model's degrees of freedom so the same instruction produces consistent, parseable results across hundreds of runs.

## Ground It
Deconstruct a task spec into its structural components: role definition, objective statement, context block, constraints/guardrails, input schema, and output schema. Show what happens when each component is omitted — demonstrate failure modes.

## Build It
Write a complete task spec as a Python dictionary, execute it against the Claude API (or a mock that prints the formatted prompt), and print the structured output to confirm the spec produces parseable JSON every time. Exercise hooks: easy — modify a single constraint in a provided spec and observe output change; medium — write a spec from scratch for a new task type; hard — build a spec validator that checks for missing components before execution.

## Use It
This is the pattern behind Claygent configurations and Clay enrichment prompts: every "Enrich" or "Research" column is a task spec with defined inputs (company URL, LinkedIn slug) and a forced output schema. Map the task spec anatomy directly to Clay's prompt configuration fields. [CITATION NEEDED — concept: Claygent task spec structure and field mapping]

## Ship It
Cover spec versioning (what changed between v1 and v2), regression testing (run 20 inputs through both versions, compare output schemas), and the cost of over-constraining (specs so tight the model can't handle edge cases). Exercise hooks: easy — bump a spec version and log the diff; medium — write a regression test comparing two spec versions on a shared input set; hard — build a spec that gracefully handles inputs that fall outside its training distribution.

## Extend It
Pointers to chained task specs (output of spec A feeds input of spec B), conditional branching within specs, and the tradeoff between single monolithic specs vs. narrow composable ones. Leave with a reading list: OpenAI's function calling docs, Anthropic's tool use guide, and the `instructor` library for typed LLM outputs.