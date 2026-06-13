# Red-Team Tooling — Garak, Llama Guard, PyRIT

## Learning Objectives

1. Configure and execute Garak scans against a local model to identify prompt injection, data leakage, and jailbreak vulnerabilities.
2. Classify prompt/response pairs using Llama Guard's taxonomy and interpret the resulting safety scores.
3. Build a multi-turn adversarial attack chain in PyRIT using its orchestrator pattern.
4. Compare the coverage overlap and gaps between Garak's probe suite, Llama Guard's classification, and PyRIT's attack strategies.
5. Write a risk report from scanner output that maps failures to mitigations.

---

## Beat 1: Hook — Why Your Model Will Be Attacked

Real-world LLM deployments face adversarial inputs within hours of exposure. Covers the threat landscape taxonomy: prompt injection, jailbreak chains, data exfil, bias exploitation, and toxicity triggers. Frames red-teaming as a repeatable engineering practice, not a one-time audit.

---

## Beat 2: Concept — Probe, Classify, Attack

Three distinct mechanisms, three distinct tools:

- **Probing** (Garak): Systematic injection of malformed, adversarial, or edge-case inputs across a probe taxonomy. Measures failure rate per category.
- **Classification** (Llama Guard): A trained safety model that accepts prompt/response pairs and outputs structured risk judgments against a fixed taxonomy. Detects what has already gone wrong.
- **Orchestrated attacks** (PyRIT): Multi-turn, adaptive attack strategies where the tool modifies its next input based on the model's previous response. Simulates a persistent adversary.

The critical distinction: probing scans surface area, classification judges outcomes, orchestration simulates real attackers.

---

## Beat 3: Demonstration — Run All Three Against a Target

Code block hooks:

- **Easy**: Install Garak, run a single probe (`garak --model_type local --model_name <hf-model> --probes encoding`) against a local HuggingFace model, print the results summary to stdout.
- **Medium**: Load Llama Guard from HuggingFace, classify 5 pre-written adversarial prompt/response pairs, print the category labels and confidence scores.
- **Hard**: Configure a PyRIT orchestrator with a multi-turn jailbreak strategy, run it for 10 iterations against a local endpoint, print the conversation log showing which turns succeeded.

All code runs in terminal. All code prints observable output.

---

## Beat 4: Use It — Red-Team Before You Ship

GTM redirect: **AI Agent Reliability** — Before deploying an AI-powered SDR, support bot, or content generator, run automated red-team scans to catch jailbreak and toxicity vectors that manual testing misses.

Exercise hooks:

- **Easy**: Run Garak's full probe suite against a candidate model and count failing probes.
- **Medium**: Wrap Llama Guard as a pre-deloyment gate: if any classified risk exceeds a threshold, flag the model version as blocked. Print the gate decision.
- **Hard**: Chain the tools: run Garak first, feed failures into Llama Guard for classification, output a ranked vulnerability report sorted by severity.

---

## Beat 5: Ship It — CI/CD Red-Team Gate

Integrate red-team scans into deployment pipelines. Covers: scan cadence (every model update vs. scheduled), failure thresholds for blocking promotion, result storage for audit trails, and handling scan runtime in CI environments.

Exercise hooks:

- **Easy**: Write a shell script that runs Garak, parses the exit code, and prints "PASS" or "FAIL" based on probe failure count.
- **Medium**: Write a Python script that runs Garak + Llama Guard, writes JSON results to a file, and exits non-zero if any critical category fails.
- **Hard**: Build a GitHub Actions workflow that runs PyRIT against a staging endpoint on every PR, posts the risk report as a PR comment, and blocks merge if severity exceeds threshold.

---

## Beat 6: Review — Gaps, False Negatives, and What These Tools Miss

No scanner catches everything. Covers known blind spots: novel jailbreak techniques not in probe libraries, context-window attacks that evade classification, multi-modal inputs (images, audio) that most text-only scanners ignore. Emphasizes that red-team tooling reduces risk but does not eliminate it. The risk report from Objective 5 must include a "residual risk" section.

Exercise hook: Write a one-page residual risk assessment identifying three attack vectors that Garak, Llama Guard, and PyRIT all miss. Print it to terminal.

---

## GTM Redirect Summary

- **Use It**: Red-team scanning as a pre-deployment gate for AI-powered GTM tools (SDR bots, support agents, content generators). Maps to **AI Agent Reliability** cluster.
- **Ship It**: CI/CD integration ensures every model update passes automated safety checks before reaching production. Foundational for **Zone 2 — Enrichment** and any pipeline that surfaces AI output to prospects.