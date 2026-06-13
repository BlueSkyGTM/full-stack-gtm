# Multi-Agent Debate and Collaboration

## Hook (Beat 1)
A single agent answering a question is a monologue — it confidently hallucinates because nothing pushes back. Multi-agent debate forces LLMs to defend, critique, and revise their outputs through structured opposition, producing more reliable reasoning at the cost of latency and tokens.

## Concept (Beat 2)
Three core mechanisms: **debate** (N agents generate, then critique each other's answers over K rounds before converging), **collaboration** (a router decomposes a task into subtasks, assigns each to a specialist agent, then merges results), and **suppression** (agents can escalate to humans or veto downstream actions when confidence falls below threshold). The key variable is communication topology — who talks to whom, in what order, and who decides when the loop stops.

## Demo (Beat 3)
Build a terminal-runnable multi-agent debate loop where three Claude instances argue over a classification task (e.g., "Should this lead be qualified?"). Each agent receives the previous round's answers, generates a critique, and submits a revised position. A judge function checks for consensus or stops at max rounds and returns the majority answer with confidence scores.

## Use It (Beat 4)
Wire the debate mechanism into a GTM agent squad: a task router receives a lead event, routes enrichment to one agent, LinkedIn research to another, and sequence drafting to a third — with a critic agent that reviews the draft before send. This maps directly to the multi-agent orchestration cluster (Agent Stack, GTM Topic 10) where the router is the SDR orchestrator and suppression gates prevent messaging unqualified leads.

## Ship It (Beat 5)
**Easy:** Add a two-agent critic loop to an existing single-prompt enrichment workflow — one agent drafts, one critiques, output ships only when critic approves.
**Medium:** Implement a 3-agent debate over lead qualification with explicit confidence scoring and human escalation when agents disagree after 3 rounds.
**Hard:** Build a full agent squad with a router node that decomposes an inbound lead into parallel subtasks (enrichment, research, draft), runs a critic pass, and either sends or escalates — all logged to a local SQLite instance for traceability.

## Evaluate (Beat 6)
Explain why debate improves output reliability and under what conditions it fails (latency-sensitive paths, low-stakes decisions where cost exceeds value). Compare the token cost of a single-agent call versus a 3-agent, 3-round debate. Identify which GTM decisions justify multi-agent overhead (qualification, outbound messaging) versus which don't (simple field mapping, data formatting).

---

**GTM Redirect:** Multi-agent orchestration (Cluster 10) — agent squad pattern with task router, specialist agents, critic node, and suppression/escalation gates. This is the autonomous SDR stack: one agent enriches, one researches, one drafts, one vets.