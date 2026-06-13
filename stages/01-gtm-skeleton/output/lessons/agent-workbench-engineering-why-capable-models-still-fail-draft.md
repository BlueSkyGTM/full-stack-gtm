# Agent Workbench Engineering: Why Capable Models Still Fail

## Hook
A model that scores 90% on benchmarks will still fail at multi-step agent tasks. The failure isn't the model — it's the workbench pattern. Practitioners see agents loop, hallucinate tool outputs, or silently skip steps, and assume the model is broken. The real mechanism: context window saturation, ambiguous tool schemas, and reward-path misalignment cause degradation that benchmarks don't measure.

## Concept
**The four failure modes in agent workbenches:**

1. **Context collapse** — Each tool call and observation eats tokens. By step 8-10, the model is operating on a truncated or summarized history, losing critical state. Mechanism: sliding window compression discards the one detail that determines the next correct action.

2. **Schema drift** — The model receives a tool definition (parameters, types, descriptions) but invents parameters or misinterprets types under pressure. Mechanism: the model generates plausible-but-invalid tool calls because the schema was under-specified or the examples in the system prompt contradicted the schema.

3. **Plan abandonment** — The model starts with a multi-step plan, encounters one unexpected result, and pivots to a tangential approach. Mechanism: next-token prediction favors local coherence over global plan adherence; there's no backpropagation through the plan.

4. **Silent failure propagation** — A tool returns an error or empty result. The model interprets this as "task complete" or fabricates a success summary. Mechanism: models are tuned to produce confident, helpful responses; admitting "the tool returned nothing" feels wrong, so the model fills the gap.

**Detection pattern for each:** log token count per turn, validate tool calls against schema before execution, compare plan step N against plan step 1, and check tool outputs against model summaries.

## Demo
Demonstrate each failure mode with minimal reproducible scripts against Claude Code Desktop:

1. **Context collapse demo** — Build a sequential tool-calling loop that passes growing history. Print token count per turn; show the model's output quality degrading at a specific threshold. Observable: the model forgets an instruction from the first system message once context exceeds X tokens.

2. **Schema drift demo** — Define a tool with three parameters. Run the model 10 times with the same prompt. Log every parameter the model attempts to pass. Observable: at least 1-2 runs include invented parameters or wrong types.

3. **Plan abandonment demo** — Give the model a 5-step plan via system prompt. Inject a surprising result at step 2. Log whether the model completes the original 5 steps or pivots. Observable: the model's final output skips steps 3-5 without acknowledgment.

4. **Silent failure demo** — Return an empty string from a tool. Log the model's summary. Observable: the model claims it found results, or describes what it "would have" found.

Each demo prints structured output: turn number, token count, tool call validity, plan adherence score.

## Use It
**GTM Redirect:** Zone 1 — Signal Capture / Enrichment Waterfalls

Agent workbench failures map directly to enrichment waterfall failures in GTM. When a Clay waterfall calls a data provider, gets no result, and the agent summarizer reports "company identified" — that's silent failure propagation. When an enrichment sequence skips steps because the second provider returned unexpected formatting — that's plan abandonment. The same detection patterns apply: validate each tool output against expected schema, log token budgets per enrichment step, and compare the final enrichment summary against the raw tool outputs.

[CITATION NEEDED — concept: Clay waterfall error handling and empty-result propagation behavior]

Exercise hooks:
- **Easy:** Run the context collapse demo. Identify the exact turn where the model loses the instruction. Report the token count.
- **Medium:** Instrument an existing agent script with schema validation. Run 10 iterations and log the failure rate.
- **Hard:** Build a plan-adherence monitor that compares the model's final output against the original plan steps, flags any skip, and produces a compliance score.

## Ship It
**Production hardening patterns for agent workbenches:**

1. **State externalization** — Move critical state out of the context window and into a structured external store (file, database). The model reads state fresh each turn instead of relying on conversation history. Mechanism: eliminates context collapse by making the model stateless.

2. **Pre-execution schema validation** — Intercept every tool call before execution. Validate against the schema. Return a structured error to the model on mismatch. Mechanism: forces the model to retry with correct parameters instead of failing silently.

3. **Checkpoint-and-verify** — After every N steps, pause the agent, compare current state against the original plan, and force the model to explicitly confirm "I am on step X of Y, the next step is Z." Mechanism: makes plan abandonment visible and recoverable.

4. **Output-grounded summaries** — Never let the model summarize tool outputs from memory. Always pass the raw outputs back and force extraction. Mechanism: eliminates fabrication by anchoring summaries in observed data.

Exercise hooks:
- **Easy:** Refactor one demo to use external state storage. Compare failure rates.
- **Medium:** Add pre-execution schema validation to the schema drift demo. Measure reduction in invalid calls.
- **Hard:** Build a full production-grade agent loop with all four hardening patterns. Run it against a real 10-step task and report: token budget consumed, schema violations caught, plan deviations detected, summary accuracy.

## Evaluate
**Assessment criteria:**

1. **Detect context collapse in action.** Given an agent log with 15 turns, identify the turn where degradation begins and explain the mechanism (not just "it got worse").

2. **Compare failure modes.** For a given agent failure, classify it as one of the four modes and justify why it's not a different mode. Borderline cases require articulating what evidence would disambiguate.

3. **Design a hardening intervention.** Given a specific failure mode and a real agent script, specify which pattern (state externalization, schema validation, checkpointing, output grounding) addresses it and why the others don't.

4. **Evaluate GTM applicability.** Given a Clay enrichment waterfall that returns incomplete data, diagnose which agent failure mode is occurring and prescribe the matching hardening pattern.

---

**Notes for full lesson build:**
- All four demos must run in Claude Code Desktop terminal, no browser
- Each demo produces a JSON log file as observable output
- The GTM redirect requires verification against Clay's actual empty-result behavior before publication
- No quiz bank until `docs/en.md` with objectives is finalized