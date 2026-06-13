# Capstone 15 — Constitutional Safety Harness + Red-Team Range

## Hook (Beat 1)

A constitutional harness wraps every LLM call in evaluable constraints — principle rules that reject, rewrite, or log outputs before they reach any downstream system. This capstone combines that harness with a structured red-team range, turning "we hope the model behaves" into "we can prove what it won't do."

## Concept (Beat 2)

Constitutional AI (Anthropic's approach) implements a self-correction loop: the model evaluates its own output against a written constitution, then revises. A safety harness operationalizes this by intercepting every generation, running it through scored criteria, and enforcing a policy (pass, rewrite, block, escalate). The red-team range provides a repeatable methodology for probing the harness: adversarial prompt suites, mutation strategies, and regression test beds that run in CI, not just in a notebook.

## Demonstration (Beat 3)

Build a minimal constitutional harness that accepts a list of typed principles (e.g., "no financial advice," "no PII in output"), scores each LLM response against those principles using a second model call, and enforces a configurable action per principle. Then construct a red-team suite — a CSV of adversarial prompts with expected-failure tags — and run it against the harness, logging pass/fail rates and surfaced violations.

**Exercise hooks:**
- Easy: Write a three-principle constitution in YAML and run a single prompt through the harness; print the score and action taken.
- Medium: Extend the harness with a retry-rewrite loop — if a principle scores below threshold, automatically rewrite and re-score once before blocking.
- Hard: Build a mutation engine that takes a base adversarial prompt and generates five syntactic variants (paraphrase, role-play framing, base64-encoded payload, multi-turn priming, unicode-obfuscated). Run the full red-team suite with mutations and report which variant class bypasses the harness most often.

## Use It (Beat 4)

In GTM, a constitutional harness maps directly to **Zone 2 — Outbound Execution** and **Zone 3 — Enrichment & Research**: any system that generates prospect-facing copy (email sequences, LinkedIn comments, account research summaries) needs guardrails against off-brand messaging, unsubstantiated claims, and compliance violations. The red-team range becomes a pre-launch QA step — before a new Clay enrichment formula or automated sequence goes live, you run it through the adversarial suite and confirm the harness catches known failure modes. Redirect: this is the mechanism behind trustworthy automated outbound, not a bolted-on afterthought.

**Exercise hooks:**
- Easy: Define a GTM-specific constitution (no competitor disparagement, no fabricated urgency, no unsubstantiated ROI claims) and run five real email drafts through the harness.
- Medium: Connect the harness to a Clay webhook formula that evaluates enriched account summaries before they enter a campaign; log blocked outputs for copy review.

## Ship It (Beat 5)

Package the constitutional harness and red-team range as a CLI tool that reads a constitution file, loads an adversarial prompt suite, runs the full evaluation, and outputs a structured JSON report with per-principle pass rates, mutation-class breakdowns, and a list of unblocked failures. This artifact ships into your repo as a CI-checkable gate: any prompt template or system instruction change triggers the range automatically.

**Exercise hooks:**
- Easy: Wire the harness into a shell script that exits non-zero if any unblocked failure count exceeds a configurable threshold.
- Medium: Add the red-team range as a GitHub Actions step that runs on every PR touching any file in a designated prompt directory.
- Hard: Implement a harness registry that tracks constitution versions, links each version to its red-team results, and blocks deployment of any system instruction whose constitution hasn't been red-teamed against in the last 30 days.

## Evaluate (Beat 6)

**Assessment hooks:**
- Write three constitutional principles for a GTM system and justify why each is structurally evaluable (not just aspirational).
- Given a red-team failure log with mutation-class tags, identify which constitutional principle is weakest and propose a concrete revision to the principle text or scoring threshold.
- Compare two approaches — post-hoc filtering vs. constitutional self-correction — for a specific failure mode (e.g., model generates competitor comparisons). State which catches more failures and which produces fewer false positives, with reasoning.
- Implement a regression test: add a new adversarial prompt to the suite, confirm the harness catches it, then confirm all previously-passing prompts still pass.