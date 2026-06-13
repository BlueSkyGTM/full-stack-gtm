# Verification Gates

## GTM Redirect Rules

This lesson maps to the **Enrichment Waterfall** cluster in Zone 2. The redirect is specific: verification gates are the boolean predicates you insert between waterfall steps to reject low-confidence enrichment before it pollutes downstream records. If you build Clay waterfalls, you build verification gates — they are the same mechanism.

---

## Beat 1: Hook — The Expensive Garbage Problem

A waterfall that enriches 10,000 accounts through three providers but accepts every result without inspection has a name: a garbage pipeline. Verification gates are the boolean filters that reject bad enrichment output before it reaches your CRM. One missing gate can corrupt a contact database for months. This lesson shows you how to build them, test them, and place them correctly.

---

## Beat 2: Concept — Predicates as Pipeline Control Flow

Explain the mechanism: a verification gate is a pure function that takes an enrichment result and returns `True` (accept) or `False` (reject). Three gate types: format gates (does the email match RFC 5322?), plausibility gates (does the employee count fall within a credible range for this company tier?), and consistency gates (does the returned domain match the input company name?). Gates compose with AND logic — all must pass. A rejected result triggers the next waterfall step; a passed result short-circuits the waterfall. This is control flow disguised as validation.

---

## Beat 3: Demo — Observable Gate Logic in Python

Show a working implementation: define three gate functions (format check on email via regex, plausibility check on employee count against a threshold, consistency check comparing domain to company name). Run a batch of five mock enrichment results through the gates. Print which results pass, which fail, and which gate caused each failure. Every line produces terminal output — no ambiguity about what happened.

---

## Beat 4: Use It — Inserting Gates into a Clay Waterfall

Map the concept directly to Clay's waterfall mechanism. Show where the gate logic lives: in formula columns that evaluate enrichment output before the next waterfall step runs. Concrete example: enrich company revenue from Provider A → gate checks `revenue > 0 AND revenue < 10T` → if False, waterfall falls through to Provider B. Explain that Clay's conditional run column IS a verification gate — it is a boolean predicate controlling whether a subsequent enrichment step executes. The mechanism is identical to the Python demo; only the interface changes.

**Exercise hooks:**
- Easy: Write a format gate that rejects any email lacking a `@` symbol. Test against five inputs.
- Medium: Build a composite gate (AND of three sub-gates) for person enrichment: email format, title keyword match, company domain consistency. Print pass/fail per record with the rejecting gate named.
- Hard: Implement a mini-waterfall simulator: three enrichment functions (mock providers), two verification gates between them, fallthrough logic, and a final report showing which provider satisfied each record — or which records failed all providers.

---

## Beat 5: Ship It — Production Verification Patterns

Describe the three patterns that survive production: gate-before-write (verify before CRM sync), gate-between-providers (verify between waterfall steps), and gate-after-merge (verify deduplicated records). Warn about the failure mode that matters: over-restrictive gates that reject valid results and drive waterfall costs through unnecessary fallthrough. Show how to instrument gate rejection rates — if a gate rejects >40% of results from a provider, the provider is the problem, not the gate.

**Exercise hooks:**
- Easy: Add a rejection-rate counter to the Beat 4 waterfall simulator. Print the rate per gate.
- Medium: Implement the gate-before-write pattern: enrich ten records, run through gates, write only passing records to a JSON file, print the rejection count and the filenames of accepted vs. rejected batches.
- Hard: Build a gate-tuning loop: run 100 mock records through a gate with a configurable threshold (e.g., minimum confidence score), log acceptance rate at five threshold values, and print which threshold produces the optimal balance (acceptance rate closest to 80%).

---

## Beat 6: Review — What You Build Next

Recap the mechanism: verification gates are boolean predicates that control waterfall fallthrough and prevent corrupt data from reaching systems of record. Three types (format, plausibility, consistency), three placement patterns (before-write, between-providers, after-merge), one warning (gate your providers, not just your data). Next lesson: confidence scoring — when a binary pass/fail gate is too crude and you need a probability instead.

**Assessment hooks:**
- Write a gate that rejects any company enrichment where returned employee count differs from input employee count by more than 50%. Print the rejection reason for each failed record.
- Explain in one sentence why placing a verification gate AFTER CRM sync defeats its purpose.
- Given a waterfall with three providers and a 30% rejection rate at Gate 1, calculate how many records reach Provider 3 out of an input batch of 1,000.

---

## Learning Objectives

1. Implement format, plausibility, and consistency verification gates as composable boolean functions.
2. Configure conditional logic between enrichment waterfall steps to reject low-quality output.
3. Diagnose over-restrictive gates by measuring and interpreting rejection rates per provider.
4. Evaluate three gate placement patterns (before-write, between-providers, after-merge) for a given pipeline architecture.
5. Build a waterfall simulator with verification gates that reports which provider satisfied each record.