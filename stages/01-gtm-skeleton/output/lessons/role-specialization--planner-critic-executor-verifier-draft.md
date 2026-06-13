# Role Specialization — Planner, Critic, Executor, Verifier

## Hook It
Multi-step LLM pipelines collapse when a single prompt tries to plan, execute, evaluate, and validate simultaneously. Role specialization splits these conflicting objectives into discrete pipeline stages, each with its own system prompt and success criteria.

## Map It
Covers the four-role decomposition pattern: Planner decomposes goals into ordered steps, Critic evaluates plans before execution, Executor runs steps in isolation, Verifier checks final output against the original goal. Includes decision framework for when to use all four vs. subset (e.g., Planner+Executor only for low-stakes tasks).

## Build It
Implement a four-role pipeline in Python that processes a task through all stages sequentially, with each role's prompt and output logged. Observable output: printed plan, critique, execution result, and verification verdict. Exercise hooks: Easy — run the pipeline on a fixed task and read the stage outputs; Medium — modify the Critic prompt to catch a specific failure mode; Hard — add a retry loop where Critic rejection triggers Planner re-planning.

## Use It
GTM redirect: This is the multi-agent decomposition behind Clay's research waterfall and smart lead scoring [CITATION NEEDED — concept: Clay agent role separation]. In GTM context, Planner decomposes an account research goal (find tech stack, funding, buying signals), Critic rejects low-confidence sources, Executor runs enrichment lookups, Verifier confirms minimum data completeness before writing to CRM.

## Ship It
Production considerations for role-specialized pipelines: token cost accumulation across stages, latency compounding (each stage is a separate API call), when to skip Critic/Verifier for speed, monitoring per-stage failure rates, and fallback behavior when Verifier rejects. Exercise hooks: Easy — add timing instrumentation to each stage; Medium — implement a budget check that skips Verifier if Critic score exceeds threshold; Hard — build a stage-level retry with max attempts and escalate-to-human on exhaustion.

## Bite It
Assessment hooks targeting: identify which role handles a given responsibility; predict pipeline behavior when one role is removed; debug a failing pipeline given stage outputs; design role assignments for a novel multi-step task. Each question maps to a specific learning objective from `docs/en.md`.