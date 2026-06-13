# Initialization Scripts for Agents

## Hook
Agents that boot without explicit initialization inherit whatever defaults the runtime provides — which means inconsistent behavior across runs, environments, and practitioners. Initialization scripts are the deterministic startup sequence that makes an agent reproducible.

## Concept
An initialization script is a deterministic function that runs once before the agent's first inference turn. It loads configuration, sets system prompts, registers tools, seeds memory/state, and validates the environment. Without it, you're relying on implicit conventions that break when you share the agent or deploy it.

## Mechanism
The boot sequence follows a strict order: environment validation → configuration load → system prompt assembly → tool registration → state seeding → readiness check. Each step can fail, and the script must fail loudly (not silently) so you know *which* step broke. The mechanism is a pure function: same inputs → same agent state, every time.

## Code
Build a complete initialization script that takes a YAML config, assembles a system prompt from modular sections, registers a tool list, seeds short-term memory with context, and prints the full agent state to confirm boot success. [CITATION NEEDED — concept: standard agent boot sequence pattern in Claude Code / Agent SDK]

## Use It
This is the mechanism behind Zone 01 (ICP Definition) deployment in GTM: when you initialize an agent with a loaded ICP definition, company metadata, and scoring thresholds at boot, every subsequent inference turn operates within those guardrails without re-prompting. The Clay waterfall enrichment pattern uses the same initialization pattern — load the enrichment cascade config once, then execute against it.

## Ship It
Package the initialization script as a standalone module that accepts a config path as a CLI argument, validates all required fields, and either returns a fully-initialized agent object or exits with a specific error code and message.

---

**Exercise Hooks:**

- **Easy:** Write an initialization script that loads a JSON config file and prints the resolved system prompt.
- **Medium:** Add tool registration and environment validation; the script must fail with a clear error if an API key is missing.
- **Hard:** Build a boot sequence that merges multiple config layers (default → environment → CLI override), registers tools dynamically from a directory, and outputs a checksum of the final agent state for reproducibility verification.

---

**Learning Objectives (draft):**

1. Build an initialization script that loads configuration and produces a deterministic agent state.
2. Implement environment validation that fails loudly with actionable error messages.
3. Construct a system prompt from modular components at boot time.
4. Compare implicit agent defaults vs. explicit initialization across multiple runs.
5. Configure an agent boot sequence that accepts external config and produces observable startup output.