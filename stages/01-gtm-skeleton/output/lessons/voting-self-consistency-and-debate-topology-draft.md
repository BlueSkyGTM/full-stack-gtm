# Voting, Self-Consistency, and Debate Topology

## 1. Hook

Single-shot LLM outputs drift on repeat calls. When classification accuracy matters — whether a company fits your ICP, whether a support ticket is escalation-worthy — you need an aggregation strategy, not a bigger prompt.

## 2. Mechanism

Three patterns for extracting a stable answer from multiple LLM samples. **Majority voting**: N independent completions, mode wins. **Self-consistency** (Wang et al. 2022): N chain-of-thought reasoning paths, extract the final answer from each, vote on answers — the insight being that correct reasoning paths converge on the same answer. **Debate topology**: M agents generate answers, then argue in rounds; a judge (or final voter) picks. Topology shapes convergence — star (all talk to one judge), pairwise (two debaters + judge), tournament (bracketed elimination). All three trade cost (more tokens) for reliability, but they fail differently. [CITATION NEEDED — concept: debate topology convergence bounds]

## 3. Implementation

Build all three aggregation methods against a shared classification task (e.g., "is this company enterprise or SMB?"). Each method receives the same input, produces a final answer, and prints the vote distribution, total token cost, and agreement rate. Observable output shows where the methods agree and diverge.

## 4. Use It

GTM redirect: **Zone 2 — ICP qualification confidence**. When a single LLM call classifies an account as ICP-fit or ICP-unfit, one bad completion flips the record. Self-consistency over 5 reasoning paths, with a vote threshold of 4/5, is the mechanism behind high-confidence enrichment. In Clay, this pattern appears when you waterfall through multiple enrichment sources and cross-validate — the voting mechanism is what turns noisy signals into a usable score. If the AI concept is purely foundational (debate topology has no clean GTM application yet), the redirect is: "foundational for Zone 3 agent architectures where multiple sub-agents must reconcile conflicting signal interpretations."

## 5. Ship It

**Easy**: Run self-consistency on 10 account descriptions, threshold at 3/5 agreement, print pass/fail with vote breakdown. **Medium**: Compare voting vs. self-consistency on the same dataset; output a CSV with columns `account_id, voting_label, sc_label, voting_cost, sc_cost, agreement` and flag rows where methods disagree. **Hard**: Implement a 3-agent debate topology with a judge. Each agent argues for a different ICP tier; the judge must cite which arguments were persuasive. Print the full debate transcript and final ruling. All exercises run in terminal with printed output — no browser.

## 6. Debug It

Common failure modes: tied votes with even N (always use odd), self-consistency extracting the wrong "final answer" from a rambling chain-of-thought, debate agents collapsing into agreement too early because the judge prompt is leading. Each mode has a signature in the vote distribution — the exercise is to read the distribution and diagnose which failure occurred.

---

**Learning Objectives** (for `docs/en.md`):
1. Implement majority voting over N LLM completions and compute agreement rates from observable output.
2. Implement self-consistency by extracting final answers from N chain-of-thought paths and voting on extracted answers.
3. Compare voting and self-consistency on the same classification task, identifying cases where they diverge.
4. Configure a multi-agent debate topology (star, pairwise, or tournament) and identify which topology structure is in use from the interaction logs.
5. Diagnose aggregation failures (tie votes, early collapse, answer extraction errors) from vote distributions.