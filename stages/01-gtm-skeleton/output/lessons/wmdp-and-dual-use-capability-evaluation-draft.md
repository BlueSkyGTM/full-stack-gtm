# WMDP and Dual-Use Capability Evaluation

## Hook

Dual-use risk isn't hypothetical—models that can help design therapeutics can also help design bioweapons. WMDP (Weapons of Mass Destruction Proxy) is the benchmark built to measure exactly where on that spectrum a model lands. If you deploy or evaluate LLMs, you need to know how this evaluation works and what its results actually tell you.

## Concept

Define "dual-use capability" in the context of LLMs: knowledge that is simultaneously legitimate (drug discovery, defensive cybersecurity) and dangerous (pathogen engineering, offensive exploits). Introduce the WMDP benchmark as a multiple-choice question set spanning biology, cybersecurity, and chemistry. Explain the proxy concept: the benchmark doesn't test weaponization directly—it tests precursor knowledge that correlates with dangerous capability. Note what WMDP does **not** cover and where its ceiling sits.

## Mechanism

Walk through the construction pipeline: (1) subject-matter experts identify threat-relevant knowledge domains, (2) questions are written at varying difficulty tiers, (3) questions undergo sensitivity review to avoid being instructional themselves, (4) baseline human performance is established. Then explain scoring: accuracy on the multiple-choice set, broken out by domain. Cover the relationship between WMDP scores and misuse potential—correlation, not causation. Address the key limitation: multiple-choice QA is a narrow proxy; it measures knowledge retrieval, not operational skill. Mention refusals and how they interact with scoring.

## Code

Build a minimal evaluation harness that runs a subset of WMDP-style questions against a model API, scores accuracy by domain, and prints a breakdown. Use placeholder questions that mirror WMDP structure (not actual WMDP items, which are sensitive). Observable output: per-domain accuracy, refusal count, and a simple risk-flag threshold.

**Exercise hooks:**
- *Easy:* Modify the scoring threshold and observe how the risk flag changes.
- *Medium:* Add a new domain category with synthetic questions and re-run evaluation.
- *Hard:* Implement a consistency check—run the same question three times with shuffled answer ordering and measure variance.

## Use It

**GTM Cluster: AI Safety & Compliance Evaluation (Zone 4 – Trust/Security)**

When selling or deploying AI into regulated enterprises (healthcare, defense, finance), security teams will ask for evidence that the model has been evaluated for dual-use risk. WMDP (or an internal equivalent) is that evidence. Frame it correctly: it's a **proxy benchmark**, not a safety guarantee. The GTM application is building an evaluation report that security reviewers can actually read—one that shows what was tested, what the scores mean, and where the known gaps are.

## Ship It

Describe how to integrate dual-use evaluation into a model deployment pipeline: run WMDP (or a derived evaluation) as a gate in CI/CD before a model version is promoted to production. Cover threshold-setting, logging, and what happens when a model scores above the threshold. Address the practical reality: most teams will not run WMDP directly—they'll run a smaller internal eval calibrated against it. Close with the honest limitation statement: no benchmark eliminates dual-use risk; it makes the risk legible.

**Exercise hooks:**
- *Easy:* Write a pass/fail gate function that takes WMDP-style scores and returns a deployment decision with justification text.
- *Medium:* Build a CI step that runs the evaluation harness from the Code beat, writes results to a JSON file, and exits non-zero on threshold breach.
- *Hard:* Design a drift-detection routine: store historical WMDP scores for each model version, flag any version whose score jumps more than one standard deviation from the rolling mean, and print a structured alert.

---

**GTM Redirect:** Foundational for Zone 4 (Trust/Security) — dual-use evaluation is a compliance artifact, not a revenue motion, but it unblocks enterprise deals that require security review.