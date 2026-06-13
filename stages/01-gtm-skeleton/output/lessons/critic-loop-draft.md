# Critic Loop

## Hook

The single-pass LLM call is a coin flip. A Critic Loop turns that coin flip into a quality gate: generate, evaluate, refine. This pattern is what separates demos from production pipelines.

## Concept

**Mechanism**: A generator produces output. A critic (often the same model with different instructions, sometimes a different model or rule set) scores that output against explicit criteria. If the score falls below threshold, the generator retries—optionally with the critic's feedback appended to the prompt. Loop terminates on pass or max iterations.

Two variants: *internal critique* (same model, different system prompt) vs. *external critique* (separate model or deterministic validator). Internal is simpler; external catches blind spots the generator shares with itself.

Key variables: iteration cap, scoring rubric specificity, feedback granularity, temperature resets between retries.

## Use It

**GTM Redirect — Enrichment Quality Gate**

[CITATION NEEDED — concept: Clay enrichment waterfall with validation step]

In a Clay waterfall, each enrichment step pulls data from a provider. A critic loop can validate the enriched field before writing it to the CRM: "Does this company description match the website we scraped? Score 1-5. If below 3, re-enrich from a different provider." This prevents garbage data from polluting downstream segments.

Relevant GTM cluster: **Data Enrichment & Scoring** (Zone 2).

## Build It

Three exercises of increasing complexity:

**Easy**: Single-model critic loop. Generator writes a one-sentence value prop for a given company. Critic scores it against a rubric (mentions pain point, under 20 words, no jargon). Loop until pass or 3 tries. Print each iteration and final result.

**Medium**: Dual-model critic. Generator uses one model (e.g., Claude Haiku for speed). Critic uses a second model or a deterministic rule set (regex + keyword check). Compare pass rates and latency between internal vs. external critique over 10 test cases.

**Hard**: Production critic loop with structured output. Generator produces a JSON enrichment record. Critic validates schema compliance + semantic quality. Track: iterations to convergence, failure modes, token cost per loop cycle. Output a summary table.

## Ship It

**Production considerations:**
- Token cost scales linearly with iterations. Cap at 3.
- Critic prompt drift: if the critic is too lenient, the loop exits on iteration 1 with bad output. If too strict, you hit the cap every time. Calibrate on 50+ samples before deploying.
- Latency: each iteration is a round-trip. For real-time GTM workflows (e.g., live chat, instant enrichment), a single-pass with post-hoc async critique may be better.
- Logging: always log every iteration, not just the final output. You need the audit trail to improve the generator prompt.

## Extension

- **Constitutional AI**: Anthropic's RLHF approach uses a specific form of critic loop where the model critiques its own outputs against a constitution. Read the original paper for the formalization.
- **Multi-critic ensembles**: Multiple critics with different rubrics vote. Useful when quality is multi-dimensional (e.g., outreach emails must be both personalized *and* compliant).
- **When to stop looping**: If your generator can't pass the critic in 2-3 tries, the problem is the prompt, not the loop count. Diagnose before iterating.