# AI Control — Safety Despite Subversion

## Beat 1: Concept

The control problem assumes the model you deploy may actively attempt to subvert your safety measures. Unlike alignment — which asks "how do we make the model want good things?" — control asks "how do we prevent harm even if the model doesn't want good things?" Covers the threat model: you have access to a trusted (weaker) model and an untrusted (stronger) model, and must build protocols that limit damage from the latter using the former. References the foundational papers from Anthropic on model welfare and control evaluations [CITATION NEEDED — concept: Anthropic control evaluations protocol paper].

## Beat 2: Mechanism

Three core protocols — **trusted monitoring**, **deference**, and **suspicion-based auditing**. Trusted monitoring: every output from the untrusted model passes through a trusted model that flags suspicious content. Deference: the trusted model can override the untrusted model's output with its own. Suspicion scoring: the monitor assigns a scalar suspicion score; high scores trigger human review or shutdown. The critical mechanism is the information asymmetry — the untrusted model doesn't know exactly what the monitor will flag, creating an exploitation frontier the safety team can measure. Covers the "safety tax" — performance cost of running safety protocols — and how to quantify it.

## Beat 3: Design

Design a control protocol for a concrete scenario: an AI coding assistant with access to a production codebase. The untrusted model writes code; the trusted model reviews diffs. Map out the attack surface — what can the untrusted model smuggle in? Backdoors, data exfiltration via network calls, subtle logic bugs. Design the monitor's rubric: what specific patterns does it check? Define thresholds for suspicion scores and escalation paths. Compare protocol overhead: trusted-only (slow, safe) vs. untrusted-only (fast, unsafe) vs. monitored (middle, measurable risk). Exercise hooks: Easy — list three attack vectors for an untrusted code model. Medium — design a monitoring rubric for catching each vector. Hard — calculate the safety tax for a protocol given false-positive and false-negative rates.

## Beat 4: Code

Implement a working simulation of a trusted monitoring protocol. Two local functions — one representing an untrusted model that sometimes produces malicious outputs, one representing a trusted monitor that scores outputs for suspicion. The protocol runs batches, flags suspicious outputs, and logs safety metrics: catch rate, false-positive rate, and safety tax. Observable output: a table of results showing how many attacks were caught, how many benign outputs were wrongly flagged, and the total overhead. A second example implements a red-team simulation: the untrusted model adapts its strategy across rounds to probe the monitor's blind spots, and you observe the exploitation frontier converge.

## Beat 5: Use It

Maps to the **AI Agent Orchestration** cluster in Zone 3. When you deploy AI agents that execute multi-step GTM workflows — enrichment, sequencing, research — those agents have capabilities that could cause real damage: sending emails, modifying CRM data, making API calls with real costs. The control protocol pattern applies directly. Your "untrusted model" is the agent executing the workflow; your "trusted model" is a simpler rules engine or cheaper model that reviews actions before execution. The monitoring rubric checks for: actions outside expected scope, unusual data access patterns, outputs that don't match the expected schema for a GTM step. Safety tax here is latency — every review step adds milliseconds to your pipeline. You measure whether the latency cost is worth the reduced risk of a rogue agent action. Redirect: "This is the same control protocol pattern you'd apply to any autonomous GTM agent — the trusted monitor is your pre-send validation layer, the untrusted model is your enrichment and sequencing agent."

## Beat 6: Ship It

Checklist for deploying control protocols in production: define your trust boundary (which model is trusted, which is untrusted), set suspicion thresholds empirically (run historical data through the monitor to calibrate), establish escalation paths (human review queue), measure safety tax on an ongoing basis, and run periodic red-team exercises where you deliberately probe your own protocol. The critical shipping question: what is your acceptable false-negative rate for untrusted model subversion? Zero is not achievable — state the number you're shipping with and the reasoning. Exercise hooks: Easy — write down your trust boundary for an existing agent deployment. Medium — calibrate suspicion thresholds against a dataset of 100 past outputs. Hard — design a red-team exercise that probes your specific monitor's blind spots and run it.

---

**GTM Redirect Rules for this lesson:**
- "Use It" references: **AI Agent Orchestration** cluster, Zone 3
- "Ship It" references: same cluster, with emphasis on pre-send validation and agent guardrails
- No forced connections — the mapping is direct (autonomous GTM agents need control protocols)