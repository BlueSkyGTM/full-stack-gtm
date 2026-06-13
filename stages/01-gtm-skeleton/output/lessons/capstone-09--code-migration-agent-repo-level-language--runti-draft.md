# Capstone 09 — Code Migration Agent (Repo-Level Language / Runtime Upgrade)

## GTM Redirect Rules

This capstone maps to **Zone 03 — Enrichment** workflows. The repo-level migration agent demonstrates the same pattern as enriching a CRM record: read source, apply transformation rules, verify output, write back. The specific mechanism is **batch transformation with verification gates**, which directly parallels Clay's enrichment waterfall where each step validates before proceeding. If the connection feels stretched, the redirect is: "foundational for Zone 03 — the agent orchestration pattern underlies all batch enrichment pipelines."

---

## Beat 1: Hook

**What a migration agent actually solves.** You have a 200-file Python 3.8 codebase that needs to be Python 3.12. `pyupgrade` handles syntax. It does not handle API deprecations, moved imports, or behavioral changes in the standard library. A migration agent reads each file, applies a rule set, runs tests, and either commits or rolls back — the same loop a human runs manually, except it doesn't get tired at file 47.

---

## Beat 2: Concept

**The migration agent loop: parse → transform → verify → commit.** This beat covers the four-phase architecture. Phase 1 parses the repo into an AST or structured representation. Phase 2 applies transformation rules (hardcoded for deterministic fixes, LLM-assisted for ambiguous ones). Phase 3 runs the existing test suite or a smoke test to verify nothing broke. Phase 4 commits or queues for human review. The key mechanism is the **verification gate**: no file advances to "done" until tests pass. Without this gate, you have a search-and-replace script with an LLM inside it.

**Sub-concepts to cover:**
- Dependency graph traversal (migrate leaf nodes first)
- Rule precedence: deterministic rules run before LLM rules
- Rollback strategy: git-based, not in-memory
- Partial migration: how to handle files that fail verification

---

## Beat 3: Demo

**Walk through a working migration agent that upgrades a toy Python 3.8 codebase to 3.12.** The demo repo has 5 files with known incompatibilities: `collections.Mapping` (removed in 3.10), `datetime.utcnow()` (deprecated in 3.12), and f-string syntax changes. The agent processes each file, applies deterministic fixes first, sends ambiguous cases to the LLM, runs `pytest`, and reports results. Every step prints observable output — no magic.

**Demo structure:**
- Show the repo layout and test suite
- Show the migration config (rules + target version)
- Run the agent, show per-file output
- Show the final git diff
- Show the test results before and after

---

## Beat 4: Exercise

**Three tiers of exercise, all building on the same scaffold repo.**

- **Easy:** Write a deterministic transformation rule that replaces `collections.Mapping` with `collections.abc.Mapping`. Run the agent on a repo that contains this pattern. Confirm the diff and passing tests.

- **Medium:** Add an LLM-assisted rule for `datetime.utcnow()` → `datetime.now(timezone.utc)`. The LLM must handle cases where the return value is used in arithmetic (the timezone-aware datetime behaves differently in subtraction). Verify with a test that checks `isinstance(result, datetime)`.

- **Hard:** Implement dependency-order traversal. The repo has files A, B, C where A imports from B and B imports from C. The agent must migrate C first, verify, then B, then A. If B fails verification after migration, the agent rolls back B and C and logs the failure.

---

## Beat 5: Use It

**Where this pattern appears in GTM workflows.**

The migration agent's loop (read → transform → verify → commit) is the same loop Clay's enrichment waterfall runs: read a record from the CRM, hit an enrichment source, check if the data is valid, write it back. The verification gate is what separates a "spray and pray" enrichment run from a reliable pipeline.

**Specific connection:** If you are enriching a list of 10,000 accounts with technographic data, you are running the same four-phase loop. The "AST" is the account record schema. The "transformation rules" are your enrichment sources. The "test suite" is your validation rules (e.g., "employee count must be an integer between 1 and 10,000,000"). The "commit" is writing back to Salesforce.

**Exercise hook:** Map a Clay enrichment workflow to the four-phase migration loop. Identify which Clay steps are deterministic rules vs. LLM-assisted steps. Identify where the verification gate is (or where it should be if it's missing).

---

## Beat 6: Ship It

**Production concerns for a migration agent.**

- **Dry-run mode:** Always run with `--dry-run` first. The agent writes diffs to a directory without touching the working tree.
- **Concurrency:** Process independent files in parallel, but respect the dependency graph. Use a topological sort.
- **Token budgeting:** An LLM-assisted rule on a 500-line file costs more than a 50-line file. Set a per-file token cap and fall back to "manual review required."
- **Human-in-the-loop:** Not every transformation can be verified automatically. The agent should support a `--interactive` flag that pauses for approval on files where the LLM confidence is below a threshold.
- **Idempotency:** Running the agent twice on the same repo should produce the same result. If file X was already migrated, skip it.

**Exercise hook:** Add `--dry-run` mode to the migration agent. The agent writes proposed diffs to `./migration_output/` without modifying source files. Confirm that running the agent twice produces identical output.

---

## Learning Objectives (for reference in full doc)

1. Implement a four-phase migration loop (parse, transform, verify, commit) that processes a multi-file Python repository.
2. Write deterministic transformation rules that target specific AST node types and produce verified output.
3. Configure LLM-assisted transformation rules with token budgets and confidence thresholds for ambiguous migration cases.
4. Implement dependency-order traversal using import graph analysis to migrate files in correct topological order.
5. Add dry-run and rollback mechanisms that ensure the migration agent is safe to run on production repositories.

---

## Notes on GTM Cluster

The closest GTM cluster is **Zone 03 — Enrichment**, specifically the pattern of batch-processing records with transformation rules and verification gates. The migration agent is a general-purpose demonstration of this pattern applied to code instead of CRM records. The Clay waterfall is the specific GTM implementation: each enrichment step is a "rule," and the waterfall's "stop if filled" logic is the verification gate.

If this connection feels forced for students who don't work with Clay directly, the redirect is: "This capstone is foundational for Zone 03. The four-phase loop (parse, transform, verify, commit) is the generic pattern that Clay, n8n, and every other enrichment tool implements. Build it once from scratch and you'll recognize it everywhere."