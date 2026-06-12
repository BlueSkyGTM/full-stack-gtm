# GTM Topic Map
<!-- Source: shared/gtm-handbook-extract.md + shared/gtm-curriculum-integration.md -->
<!-- Capture date: 2026-06-12 | Git hash: 6d2e414 -->

## Outcomes Thesis

A Fullstack GTM Engineer is not an AI practitioner who happens to know sales tools. They are a compound practitioner: the AI engineering curriculum teaches the foundation; the GTM redirect shows where it lands in a revenue system. After completing this course, a student can demonstrate five practitioner outcomes:

---

### Outcome 1 — Signal Machine
**Can you find who is in-market before they raise their hand?**

Build systems that detect buying intent from public signals: job postings, funding events, executive hires, website visits, social engagement, and news events. The signal machine runs continuously without manual prospecting. Output: a scored, enriched list of accounts in motion, delivered to a sequence before competitors have time to act.

Core handbook clusters: TAM Mapping (1.1), Scraping Directories (3.1), News-Led Outbound (3.2), Inbound-Led Outbound (3.3), Job Change Playbook (4.3), Social Signals (4.4).

---

### Outcome 2 — Score and Qualify
**Can you build a list worth sending to?**

Transform a raw TAM into a ranked, validated, ICP-scored prospect database. The scoring system combines firmographic, technographic, and behavioral signals into a single ICP column that gates downstream outreach. Includes waterfall enrichment, email validation, and CRM hygiene. Output: a living database where every record is ICP-true, enriched, and validated before any outreach fires.

Core handbook clusters: TAM Refinement & ICP Scoring (1.2), Data Enrichment waterfalls (4 tools in sequence), CRM Playbook (4.2), PLG Playbook (4.1).

---

### Outcome 3 — Write at Scale
**Can you generate copy that gets replies without doing it manually for each prospect?**

Use AI personalization to write outbound copy at scale without losing the signal that makes cold email work: a specific, research-backed first line. Applies the 80/20 testing protocol (500 sends per variant, 1-in-250 threshold) to find a working value proposition before scaling. Output: a working, tested sequence — subject line, body, follow-up cadence — running at 3+ value proposition variants simultaneously.

Core handbook clusters: Copywriting & Testing (1.3), AI Personalization Frameworks (embedded in 1.3), Micro Lists & Outbound 3.0 (2.3), Deliverability & Infrastructure (1.4).

---

### Outcome 4 — Living GTM System
**Can you build revenue infrastructure that runs without constant human reset?**

Design and ship production GTM infrastructure: sending domains that don't burn, inbox management across multiple accounts, CRM enrichment on a refresh schedule, signal monitoring that surfaces events automatically. The system generates pipeline as background work. Output: an operational GTM stack with documented playbooks — the kind that commands a $10K/month retainer.

Core handbook clusters: Deliverability & Cold Email Infrastructure (1.4), Ex-Champions Playbook (4.6), Conference & Event Playbook (4.5), Outbound at Scale — Full TAM Playbook (4.7), GTM Analysis & Premium Positioning (4.8).

---

### Outcome 5 — Agent Stack
**Can you build autonomous GTM agents that research, enrich, personalize, and send without human input?**

Wire the GTM workflow into an agentic loop: task router decides what fires next, each tool call is a node (Clay → enrichment → LLM → sequence), the loop runs until the lead is qualified or disqualified. Includes LinkedIn automation at scale, multichannel sequencing, and multi-agent orchestration with explicit suppression and human escalation design. Output: an autonomous SDR agent running on your own infrastructure.

Core handbook clusters: LinkedIn Automation & ABM (2.1), Cold Calling Infrastructure (2.2), full multi-agent orchestration patterns.

---

## Zone Table

Each row shows: which AI engineering zone carries the GTM redirect, which handbook cluster it implements, which practitioner outcome it builds toward, and the redirect hook that bridges the two.

| Zone | AI Engineering Core | GTM Cluster(s) | Outcome | GTM Redirect Hook |
|-------|---------------------|---------------|---------|-------------------|
| 01 | Python, CLI, workspaces | TAM Mapping (1.1) | Signal Machine + Score & Qualify | "This Python env is where you'll run Clay webhooks and Apollo API calls" |
| 02 | Data structures, APIs, JSON | TAM Refinement & ICP Scoring (1.2) | Score & Qualify | "Every lead score is a JSON object — here's what yours will look like" |
| 03 | Web scraping, HTML parsing | Scraping Directories (3.1), News-Led Outbound (3.2) | Signal Machine | "This scraper becomes your hiring signal detector; this RSS feed surfaces funding events in real time" |
| 04 | Data pipelines, ETL | Enrichment Waterfalls (1.2 tooling) | Score & Qualify | "This is the Clay waterfall — Find → Enrich → Transform → Export" |
| 05 | LLM prompting, few-shot | Copywriting & AI Personalization (1.3), Micro Lists (2.3) | Write at Scale | "This prompt template writes your first 1,000 cold emails; this testing protocol finds the one that works" |
| 06 | Embeddings, semantic search | Inbound-Led Outbound (3.3) | Signal Machine | "This embedding model routes inbound leads to the right sequence before they go cold" |
| 07 | Fine-tuning, RLHF | ABM signal orchestration (2.1 + 4.3 + 4.4 + 4.5) | Signal Machine + Living GTM | "Fine-tuning = training your scoring model on your own deal history. Job changes, social signals, and events are your labels." |
| 08 | Vector databases, retrieval | CRM Architecture & Data Hygiene (4.2), PLG Playbook (4.1) | Score & Qualify | "Your CRM is a retrieval system — here's how to query it like one and keep it clean" |
| 09 | Agents, tool use, function calling | Workflow Automation (n8n/Make), Cold Calling Infrastructure (2.2) | Agent Stack | "You built the algorithm. Now wire it into a loop: task router decides what fires next, each tool call is a node." |
| 10 | Multi-agent orchestration | Multi-agent GTM systems (agent squad pattern) | Agent Stack | "This isn't just agents running in parallel — it's a task squad with a router. One lays bricks, one cements." |
| 11 | Evaluations, LLM testing | Revenue Intelligence (Gong, reply classification) | Living GTM | "Evals = A/B testing your sequences before they go live; reply classification is your eval feedback loop" |
| 12 | Observability, logging, tracing | Feedback loops, pipeline health monitoring | Living GTM | "This tracing setup monitors your sequence performance in real time; reply rate drift is your model degradation signal" |
| 13 | Deployment, CI/CD | Production GTM Infrastructure (1.4 at scale) | Living GTM | "This deploy pipeline ships your Clay tables and n8n workflows; SPF/DKIM/DMARC is your infrastructure layer" |
| 14 | Cost optimization, latency | GTM Stack Cost Management (Clay credits, API costs) | Living GTM | "Every Clay credit is a token cost — optimize like you would LLM calls" |
| 15 | Security, auth, compliance | Outbound security: webhook auth, CAN-SPAM, GDPR | Living GTM | "Your GTM stack has an attack surface — rotating API keys, securing webhooks, handling prospect data under GDPR" |
| 16 | Distributed systems | Enrichment waterfall concurrency, rate limits, retry logic | Living GTM | "Your enrichment waterfall is a distributed system — parallel requests, rate limit backpressure, idempotent retries" |
| 17 | MLOps, model lifecycle | GTM system lifecycle: versioning Clay tables, scoring drift, retraining | Living GTM | "MLOps for GTM = versioning your enrichment waterfalls, detecting when your scoring model drifts" |
| 18 | Advanced prompting, CoT | Advanced ABM personalization: multi-step research chains | Write at Scale + Agent Stack | "CoT prompting = how your agent reasons about an account before writing the first line" |
| 19 | RAG | Knowledge-augmented outreach: product docs, case studies in copy | Write at Scale + Agent Stack | "RAG = giving your outbound agent memory of your best customer stories" |
| 20 | AI systems design, capstone | Full GTM system: ICP to closed deal + GTM Analysis (4.8) | All 5 outcomes | "Design the full GTM system from scratch: signal machine → score → write → system → agent" |

## Cluster-to-Zone Coverage Check

| Handbook Cluster | Primary Zone(s) | Outcome |
|-----------------|-----------------|---------|
| 1.1 TAM Mapping | 01 | Signal Machine + Score |
| 1.2 TAM Refinement & ICP Scoring | 02, 04 | Score & Qualify |
| 1.3 Copywriting & Testing | 05 | Write at Scale |
| 1.4 Deliverability & Infrastructure | 13 | Living GTM |
| 2.1 LinkedIn Automation & ABM | 07, 10 | Signal Machine + Agent Stack |
| 2.2 Cold Calling Infrastructure | 09 | Agent Stack |
| 2.3 Micro Lists & Outbound 3.0 | 05 | Write at Scale |
| 3.1 Scraping Directories | 03 | Signal Machine |
| 3.2 News-Led & Signal-Based Outbound | 03 | Signal Machine |
| 3.3 Inbound-Led Outbound | 06 | Signal Machine |
| 4.1 PLG Playbook | 08 | Score & Qualify |
| 4.2 CRM Playbook | 08 | Score & Qualify |
| 4.3 Job Change Playbook | 07 | Signal Machine |
| 4.4 Social Signals Playbook | 07 | Signal Machine |
| 4.5 Conference & Event Playbook | 07 | Living GTM |
| 4.6 Ex-Champions Playbook | 07 | Living GTM |
| 4.7 Outbound at Scale | 10, 11 | Living GTM + Agent Stack |
| 4.8 GTM Analysis & Premium Positioning | 20 | All outcomes |

**All 18 clusters mapped. ✓**

## Zone overload check

| Zone | GTM concepts | Within limit? |
|-------|-------------|---------------|
| 07 | ABM, Job Change, Social Signals, Conference, Ex-Champions (5 clusters) | ⚠ Over 3 — split note: all 5 are variations on the same "signal-to-sequence" pattern; treat as one cluster family (Account Signal Playbooks) for lesson design |
| 05 | Copywriting, AI Personalization, Micro Lists (3 clusters) | ✓ exactly 3 |
| 08 | CRM, PLG (2 clusters) | ✓ |
| All others | 1-2 clusters | ✓ |

**Zone 07 split note:** The 5 playbooks in Zone 07 are all expressions of one pattern — detect a signal, enrich the account, fire a contextual sequence. Lyra should teach the pattern once and show the 5 playbooks as variations, not 5 independent concepts.
