# Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

---

## Beat 1: Hook

**Why agent SDKs exist.** Raw API calls to LLMs are stateless, single-turn, and lack persistence across interactions. Agent SDKs wrap the orchestration loop (tool call → observe → decide → repeat) into a reusable abstraction. This beat frames the problem: you have LLM access, you have tools, but wiring them together in production requires a protocol.

---

## Beat 2: Concept

**Three competing specifications for agent-tool binding.**

- **Anthropic Skills**: A JSON schema + convention for declaring tools Claude can invoke. The mechanism is function declarations passed in the `tools` array of the Messages API, combined with a stop-and-resume loop where the model yields control back to your code when it wants to call a tool.
- **AGENTS.md**: A proposed open standard (analogous to `robots.txt`) for declaring agent capabilities, permissions, and available actions on a given domain or codebase. The mechanism is a markdown file parsed by agent runtimes to determine what the agent is allowed to do without explicit per-step human approval.
- **OpenAI Apps SDK**: OpenAI's agent framework providing tool definitions, conversation threading, and a run loop. The mechanism is the Assistants API with function calling, where tools are registered as `functions` on an assistant object and executed server-side in a managed loop.

**Key distinction**: Anthropic and OpenAI both implement a "tool loop" but differ in where state lives (client-side vs. server-side). AGENTS.md is orthogonal — it's a permission spec, not an execution engine.

---

## Beat 3: Build

**Implement a minimal tool-calling agent in all three frameworks.**

Each example defines the same tool (e.g., "look up a company's tech stack"), wires it into the SDK, runs a query, and prints the tool call + final response. Observable output at every step.

- Exercise hook (easy): Run a single Anthropic tool call with a hardcoded response, print the tool invocation.
- Exercise hook (medium): Implement the full tool loop in OpenAI's SDK — assistant calls a function, your code executes it, feeds the result back, assistant produces a final answer. Print each loop iteration.
- Exercise hook (hard): Write an `AGENTS.md` file for a mock GTM codebase, then parse it with a script that validates the schema and prints the declared capabilities.

---

## Beat 4: Use It

**GTM Redirect: Zone 03 — Enrichment Agents**

[CITATION NEEDED — concept: agent SDKs applied to enrichment waterfall orchestration]

Agent SDKs map directly to the Clay waterfall pattern: call tool A (e.g., clearbit), observe the result, decide if tool B (e.g., Hunter) is needed, repeat. The SDK's tool loop is the same control flow as an enrichment waterfall. Build an enrichment agent using the OpenAI function-calling loop that chains 2–3 data providers and halts when sufficient data is collected.

- Exercise hook (medium): Wire two enrichment APIs as tools in Anthropic's format. Run a query for a domain and print the waterfall trace showing which tools were called and in what order.

---

## Beat 5: Ship It

**Production concerns for agent loops.**

- Token budgeting: tool calls consume context window. Implement a max-turns guard.
- Error handling: tools fail. The agent must receive error messages, not crash. Show the pattern of returning error strings as tool results.
- Permission enforcement via AGENTS.md: before executing a tool, check the declared permissions. Reject if not allowed.
- Cost tracking: log input/output tokens per loop iteration.

- Exercise hook (hard): Add a max-turns limit, error handling, and permission check to the OpenAI agent from Beat 3. Feed it a query that would loop infinitely without the guard. Print the termination reason.

---

## Beat 6: Extend

**Where agent SDKs are headed and what to watch.**

- Convergence: Anthropic and OpenAI are both moving toward server-side agent loops with managed state. Track the shift from client-side tool loops to hosted agents.
- AGENTS.md adoption: currently a proposal. Monitor which runtimes implement parsers for it.
- Multi-agent orchestration: SDKs currently handle single-agent tool loops. The next layer is agent-to-agent delegation. [CITATION NEEDED — concept: multi-agent SDKs for GTM pipeline orchestration]
- Security boundary: AGENTS.md is permission declaration, not enforcement. Enforcement is runtime-specific and inconsistent.