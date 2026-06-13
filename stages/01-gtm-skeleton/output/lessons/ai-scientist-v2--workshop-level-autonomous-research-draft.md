# AI Scientist v2 — Workshop-Level Autonomous Research

---

## Beat 1: Hook

The AI Scientist v1 could write a paper. V2 can run a research workshop — generating hypotheses, designing experiments, executing code, interpreting results, and iterating when experiments fail. The difference between "produces output" and "conducts research" is a closed-loop feedback mechanism with self-correction. That mechanism is what this lesson dissects.

---

## Beat 2: Concept

**The research loop as a state machine.** AI Scientist v2 implements a multi-stage pipeline where each stage gates the next: idea generation → literature review → experimental design → code execution → result analysis → manuscript drafting → automated peer review → revision. The critical upgrade from v1 is **workshop-level execution** — the system handles experiment failures, adjusts parameters, and re-runs rather than halting.

**Core concepts to cover:**
- The seven-stage autonomous research pipeline and its state transitions
- How v2 differs from v1: workshop-level autonomy vs. single-pass generation
- The role of the "experimentalist" agent in designing runnable experiments
- Error recovery as a first-class mechanism, not an afterthought
- Automated peer review as a quality gate and iteration trigger

**Key mechanism:** The system maintains a persistent experiment state. When code execution fails, the experimentalist agent reads the error, modifies the experimental plan, regenerates code, and retries — up to a bounded iteration count. This is not prompt chaining; it is a **failure-aware re-planning loop**.

[CITATION NEEDED — concept: AI Scientist v2 architecture, Sakana AI 2025]

---

## Beat 3: Mechanism

**Decompose the autonomous research loop into its component mechanisms:**

1. **Idea generation via semantic search over prior work.** The system embeds existing papers, retrieves semantically similar work, and generates novel research directions by combining underexplored intersections.

2. **Experimental design as constrained code generation.** The experimentalist agent receives a hypothesis and produces Python code that must: (a) execute without error, (b) produce measurable metrics, (c) complete within a compute budget. Constraints are enforced via a validation layer, not prompt instructions.

3. **Execution sandbox with error feedback injection.** Code runs in an isolated environment. If execution fails, the stderr output is injected directly into the experimentalist's context window for the next attempt. This is the core self-correction mechanism.

4. **Result interpretation as structured extraction.** Raw metric outputs are parsed into a structured schema (metric name, value, direction, statistical significance if applicable). This schema is what the writer agent consumes.

5. **Automated peer review as LLM-as-judge.** A separate reviewer agent scores the manuscript against a rubric. If scores fall below threshold, the system routes back to revision with specific reviewer feedback injected.

6. **Token budget governance.** Each stage has a token allocation. The system tracks cumulative spend and terminates gracefully if the budget exhausts mid-iteration.

**Skeptical note:** The paper claims "workshop-level" autonomy, but the system still requires pre-defined experiment templates and domain-specific scoring rubrics. The autonomy is constrained to the space the templates permit. The mechanism is real; the generality is bounded.

---

## Beat 4: Code

**Build a minimal autonomous research loop.** Not the full AI Scientist — that requires GPU clusters and paper corpora. Instead, implement the **core mechanism**: a hypothesis → experiment → error → re-plan → execute → interpret loop.

**Exercise hooks:**

- **Easy:** Implement a two-stage agent that generates a Python hypothesis-test, executes it, and prints pass/fail. No retry. Observe what happens when the generated code has a syntax error.

- **Medium:** Add the failure-aware re-planning loop. When execution fails, feed the error back to the code generator. Cap retries at 3. Print the retry chain — original code, error, revised code, final result.

- **Hard:** Implement the full mini-loop: idea generator → experimentalist → sandbox execution → result parser → reviewer agent (scoring on a simple rubric) → revision gate. Output a structured research report as JSON. Track token usage across stages.

All code runs in Claude Code Desktop terminal. No browser. Every example prints observable output confirming the loop executed and where it succeeded or failed.

---

## Beat 5: Use It

**GTM redirect: Automated ICP research loops.**

The autonomous research loop maps directly to Zone 1 (ICP & Signal) in the GTM topic map. Specifically:

- **Idea generation → Market signal detection.** Instead of searching paper corpora, search company databases, job postings, and funding events. Generate hypotheses about which signals predict purchase intent.
- **Experimental design → Signal validation.** Design tests: "Companies that hired a Head of Data Engineering in the last 90 days are 3x more likely to evaluate reverse ETL tools." Generate the SQL or API query to test it.
- **Error recovery → Data pipeline resilience.** When the signal query fails (API rate limit, schema mismatch), the re-planning loop adjusts the query, just like the experimentalist adjusts experiment code.
- **Peer review → Confidence scoring.** The reviewer agent becomes a confidence scorer: does this signal actually correlate with conversion, or is it noise?

**Mechanism transfer, not metaphor:** The re-planning loop is the same mechanism whether the agent is debugging a PyTorch training script or retrying a failed Crunchbase API call. The state machine is identical. The domain changes; the algorithm does not.

**Exercise hook:** Implement a signal-validation loop that tests three hypotheses about your ICP using any public data source. Use the retry mechanism from Beat 4. Output a JSON report with hypothesis, evidence, confidence score, and recommended action.

---

## Beat 6: Ship It

**Deploy a workshop-level autonomous research agent for a self-contained research question.**

The ship exercise requires:
1. Define a research question with measurable success criteria
2. Configure the full loop: idea → experiment → execution → analysis → review → revision
3. Set token/compute budgets and retry limits
4. Run the agent end-to-end
5. Produce a structured output: research question, methodology, results, limitations, confidence level

**GTM redirect (same as Beat 5):** The output format maps to the same structure a GTM team needs for market research reports — question, evidence, confidence, action. The agent is a research automation pipeline, whether the domain is ML benchmarks or market segmentation.

**Exercise hook (ship-only):** Run the full autonomous loop on a real question. Options:
- ML research: "Does learning rate warmup improve convergence on dataset X with model Y?" (requires local compute)
- GTM research: "Which intent signals most strongly correlate with enterprise SaaS purchases in the data infrastructure vertical?" (requires API access to a data source)

Output is a JSON report. The agent must handle at least one execution failure via the retry mechanism. Print the full retry chain as evidence the self-correction loop activated.

---

## Learning Objectives (for `docs/en.md`)

1. **Diagram** the seven-stage autonomous research pipeline and identify which stages gate subsequent stages.
2. **Implement** a failure-aware re-planning loop that retries failed code execution with error feedback injection.
3. **Compare** AI Scientist v1 single-pass generation vs. v2 workshop-level execution by identifying the specific mechanism that enables self-correction.
4. **Configure** token budget governance across pipeline stages and explain how budget exhaustion affects output quality.
5. **Transfer** the autonomous research loop mechanism to a GTM signal-validation pipeline, preserving the re-planning and confidence-scoring components.

---

## Notes

- This lesson requires API access to an LLM (Claude, GPT-4, or local model) for the agent calls. All exercise code assumes the student has `ANTHROPIC_API_KEY` or equivalent set as an environment variable.
- The "full AI Scientist v2" is not reproducible in a lesson — it requires clusters, paper corpora, and GPU hours. This lesson teaches the **mechanism**, which is reproducible.
- No quiz bank is written here. If quizzes are needed, they must be grounded in the objectives above and documented in `docs/en.md` per the rules.