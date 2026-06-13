# Human-in-the-Loop: Propose-Then-Commit

---

## Hook It

A propose-then-commit pattern separates AI-generated suggestions from executed actions. This beat frames the problem: AI outputs that directly mutate CRM data, send emails, or modify lead status without review create expensive cleanup. The practitioner has likely seen this already — an enrichment run that overwrote good data, or an automated outreach that went to the wrong segment. Propose-then-commit is the state machine that prevents this.

---

## Ground It

This beat explains the mechanism as a two-phase transaction. Phase one: the AI system writes proposals to a staging buffer with a `proposed` status. Phase two: a human reviews and transitions proposals to either `committed` or `rejected`. The key design decisions: what gets staged (everything vs. confidence-thresholded), who reviews (role-based gates), and what happens to stale proposals (TTL and expiry). The beat covers the tradeoff — latency vs. safety — and when the pattern is worth the overhead. Mention eventual alternatives: confidence-based auto-commit for high-certainty outputs, fallback to human review below threshold.

---

## Show It

Code example: a minimal propose-then-commit system in Python. A `ProposalStore` class accepts AI-generated proposals, stores them with metadata (timestamp, confidence, proposer, payload), and exposes a `review` method that accepts or rejects by ID. A `commit` function executes the approved action (in this case, printing what would happen — observable output). The code demonstrates the full lifecycle: propose → list pending → approve → commit, and also shows rejection and expiry of stale proposals. All output is printed to terminal.

---

## Use It

This is the Clay waterfall connection. In GTM, the waterfall pattern enriches records through multiple data providers sequentially. When AI is layered on top — scoring, personalization, routing — the outputs are proposals until a human approves them. This beat walks through the specific GTM scenario: an enrichment run proposes account tier upgrades based on firmographic signals. The practitioner reviews proposals in a queue, commits the valid ones (which sync to CRM), and rejects the rest. The redirect is explicit: this is the approval layer you add between Clay's enrichment waterfall and your CRM write-back. [CITATION NEEDED — concept: Clay proposal review queue UI or integration pattern for human approval of enrichment results]

---

## Ship It

Production considerations: audit logging (who approved what, when), proposal TTL (stale suggestions that no one reviewed), batch review workflows (reviewing 50 proposals at once instead of one-by-one), and rollback mechanics (what happens when a committed proposal was wrong). The beat also covers monitoring: tracking approval rates (if 95% get approved, the review gate is theater), rejection reasons (feeding back into prompt engineering), and latency impact on SLA-bound workflows.

---

## Prove It

Three exercise hooks:

- **Easy**: Extend the `ProposalStore` with a TTL-based expiry mechanism that automatically rejects proposals older than a configurable threshold. Print the expired proposals.
- **Medium**: Build a confidence-threshold router — proposals above 0.9 confidence auto-commit, proposals between 0.6 and 0.9 enter human review queue, proposals below 0.6 auto-reject. Print routing decisions.
- **Hard**: Implement a batch review interface that groups proposals by type, shows aggregate confidence stats, and allows accept-all / reject-all with per-item overrides. Print a summary report of committed vs. rejected proposals with reasoning.

---

## Learning Objectives (3–5)

1. **Implement** a two-phase propose-then-commit state machine with staging, review, and execution phases.
2. **Configure** confidence-based routing thresholds that determine which AI outputs require human review.
3. **Evaluate** when propose-then-commit adds meaningful safety vs. unnecessary latency in a GTM workflow.
4. **Build** audit logging and TTL expiry for a proposal queue.
5. **Explain** the relationship between propose-then-commit and the Clay enrichment waterfall's output stage.