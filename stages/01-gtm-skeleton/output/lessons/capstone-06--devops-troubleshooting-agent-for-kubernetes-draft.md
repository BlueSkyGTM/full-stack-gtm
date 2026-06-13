# Capstone 06 — DevOps Troubleshooting Agent for Kubernetes

## Learning Objectives

1. Build a multi-step agent that diagnoses Kubernetes failures by chaining `kubectl` commands through tool use
2. Implement a diagnostic decision tree that classifies pod failures (CrashLoopBackOff, ImagePullBackOff, OOMKilled, pending) and selects remediation paths
3. Configure structured output schemas for incident reports consumable by PagerDuty or Slack webhooks
4. Evaluate diagnostic accuracy against synthetic cluster failures with deterministic validation

---

## Beat 1: Hook

**The problem:** On-call engineers spend 15–45 minutes per incident gathering logs, describing pods, checking events, and correlating across resources before reaching a diagnosis. Mean-time-to-resolution (MTTR) is dominated by information gathering, not reasoning. Build an agent that automates the gathering and produces a structured diagnosis.

---

## Beat 2: Concept

**Mechanism:** The agent implements a ReAct loop over a restricted tool set (`kubectl logs`, `kubectl describe`, `kubectl get events`, `kubectl top`). Each tool invocation is a single bounded `kubectl` command. The LLM reasons over tool output to decide: invoke another tool, or produce a final diagnosis. The decision tree is encoded in the system prompt, not hardcoded — the model classifies the failure mode and selects the next diagnostic step autonomously.

**Key pattern:** Narrow tool surface > broad tool surface for reliability. Five focused `kubectl` commands outperform a generic shell executor for diagnostic accuracy and safety.

---

## Beat 3: Build It

**Build stages (cumulative):**

1. **Tool definitions** — Define five `kubectl`-based tools as functions with structured input schemas (resource type, namespace, resource name, tail length)
2. **ReAct loop** — Wire tools into an Anthropic tool-use loop that terminates on a `diagnosis` tool call with structured output
3. **Failure classifier** — Add a system prompt that maps observed symptoms to one of five failure categories and dictates which tool to call next based on the category
4. **Report generator** — Convert the final diagnosis into a structured incident report (JSON schema: failure_type, root_cause, affected_resources, remediation_steps, confidence)

**Exercise hooks:**
- *Easy:* Build the five tool definitions and verify they produce valid `kubectl` commands from structured input
- *Medium:* Implement the full ReAct loop and run it against a kind cluster with a deliberately crash-looping pod
- *Hard:* Add a confidence calibration step — run the same diagnosis three times, compare outputs, flag disagreement for human review

---

## Beat 4: Use It

**GTM Redirect:** This agent architecture is the technical foundation for **Zone 3 — Technical Buyer Enablement**. DevOps tooling companies (Datadog, Harness, HashiCorp) position AI-assisted troubleshooting as a differentiator to technical buyers (SREs, platform engineers). The mechanism — narrow tool surface, structured diagnostic output, ReAct reasoning — is the same pattern these vendors use in product demos to the infrastructure persona. If you sell to DevOps buyers, this capstone is a reference implementation of what their evaluation team will test.

**Exercise hooks:**
- *Easy:* Write a one-paragraph positioning statement for this agent that names the failure modes it handles and the persona it serves
- *Medium:* Map the five-tool surface to a feature comparison matrix against one commercial K8s troubleshooting tool

---

## Beat 5: Ship It

**Deployment considerations:**
- **RBAC:** The agent's service account needs `get`, `list`, `watch` on pods/events/deployments — never `delete` or `update`. Show the exact ClusterRole.
- **Alert ingestion:** Wire the agent to Prometheus Alertmanager webhooks so it triggers on firing alerts, not on a cron schedule
- **Output routing:** Send structured diagnoses to Slack (severity < critical) or PagerDuty (severity ≥ critical) based on the confidence and failure_type fields
- **Guardrails:** Add a token budget (max 8 tool calls per diagnosis) and a timeout (60s) to prevent runaway loops on ambiguous failures

**Exercise hooks:**
- *Easy:* Write the ClusterRole and ClusterRoleBinding YAML for the agent's service account
- *Medium:* Deploy the agent as a Kubernetes Deployment with alertmanager webhook ingestion and verify end-to-end diagnosis on a synthetic alert
- *Hard:* Add a feedback loop — store each diagnosis, let the on-call engineer mark it correct/incorrect, and surface accuracy metrics in Prometheus

---

## Beat 6: Evaluate

**Validation method:**

1. **Synthetic failure suite** — Create five deterministic failure scenarios in a `kind` cluster (CrashLoopBackOff from bad command, ImagePullBackOff from nonexistent image, OOMKilled from memory limit, Pending from insufficient resources, FailedScheduling from taints)
2. **Run agent against each** — Capture: failure classification accuracy, root cause identification, remediation correctness, tool call count, latency
3. **Accuracy threshold** — Agent must classify ≥4/5 correctly and produce actionable remediation on ≥3/5 to pass
4. **Regression** — Re-run suite after any prompt or tool change; log results to a JSONL file for comparison

**Exercise hooks:**
- *Easy:* Set up the five failure scenarios in a kind cluster and verify each produces the expected symptom
- *Medium:* Run the full evaluation suite against your agent and produce the accuracy report
- *Hard:* Implement the regression harness — a script that runs the suite, compares against the previous run's JSONL, and exits non-zero if accuracy dropped

---

## GTM Redirect Rules Summary

| Section | GTM Redirect |
|---------|-------------|
| Use It | Zone 3 — Technical Buyer Enablement (infrastructure persona) |
| Ship It | Foundational for Zone 3 — this is a reference architecture, not a GTM tactic itself |

Redirect is specific: the agent's narrow-tool ReAct pattern is the mechanism behind K8s troubleshooting features sold to SRE/platform personas. Not a fabricated connection — this is what Datadog, Harness, and k9s-style tools actually ship.