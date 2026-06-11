# GTM Curriculum Integration — Session Handoff Doc

Source: GTM_curriculum_integration.pdf

---

## Active Work State

This document records the curriculum restructure decisions for the Full-Stack GTM Engineering course. It captures how the existing 498 AI engineering lessons get redirected into GTM engineering application.

---

## Decisions Made

1. **AI engineering is the core curriculum.** The 498 existing lessons are not replaced — they are redirected. GTM is the application layer, not a parallel track.
2. **"Fullstack GTM Engineer" is the right label.** Not "AI SDR," not "RevOps Engineer." The role combines AI engineering + GTM execution as a single compound discipline.
3. **Lesson redirect principle:** Every AI engineering lesson gets a GTM world application. The lesson teaches the AI concept; the redirect hook shows where it shows up in a real GTM stack.
4. **Jobs stay separate.** The AI engineering lesson and the GTM application are distinct outputs. Do not merge them into hybrid lessons at the lesson level — that happens at the course experience layer (Helix).

---

## Standing Rules

- Do not rename existing lesson IDs — the site render pipeline depends on them.
- GTM redirect hooks are additive — they do not replace existing lesson content.
- Max 3 GTM concepts per phase to avoid scope creep per stage.
- Priority order for Phase work: 11 → 14 → 13 → 05 → 15–17.

---

## GTM Redirect Map — All 20 Phases

| Phase | AI Engineering Topic | GTM World Application | Redirect Hook |
|-------|---------------------|----------------------|---------------|
| 01 | Workspaces, Python setup, CLI fundamentals | ICP & TAM modeling; scripting your first prospect list pull | "This Python env is where you'll run Clay webhooks and Apollo API calls" |
| 02 | Data structures, APIs, JSON | Lead scoring inputs; pulling enrichment data via API | "Every lead score is a JSON object — here's what yours will look like" |
| 03 | Web scraping, HTML parsing | Signal detection; scraping job postings, LinkedIn, funding announcements | "This scraper becomes your hiring signal detector" |
| 04 | Data pipelines, ETL basics | Enrichment waterfalls; building multi-source data pipelines | "This is the Clay waterfall — Find → Enrich → Transform → Export" |
| 05 | LLM prompting, few-shot learning | Outbound copy generation; personalized sequence messaging at scale | "This prompt template writes your first 1,000 cold emails" |
| 06 | Embeddings, semantic search | Inbound lead routing; matching inbound signals to ICP definitions | "This embedding model routes inbound leads to the right sequence" |
| 07 | Fine-tuning, RLHF basics | Training a lead scoring model on closed-won CRM data; adapting model weights to your specific deal signals | "Fine-tuning = training your scoring model on your own deal history. Every closed-won and lost deal is a labeled training example." |
| 08 | Vector databases, retrieval | CRM as vector store; account memory and context retrieval | "Your CRM is a retrieval system — here's how to query it like one" |
| 09 | Agents, tool use, function calling | GTM agent loop design; build the research→enrich→personalize→send algorithm from scratch, then implement it as a task-routing agentic loop using the GTM operator kit | "You built the algorithm. Now wire it into a loop: task router decides what fires next, each tool call is a node, the loop runs until the lead is qualified or disqualified. That's your automated SDR." |
| 10 | Multi-agent orchestration | Multi-agent GTM systems with explicit task routing; design the manifest, assign agents by specialty (signal → enrichment → copy → send), implement the loop that coordinates them without human input | "This isn't just agents running in parallel — it's a task squad with a router. One lays bricks, one cements. Design the routing logic and the loop runs itself." |
| 11 | Evaluations, LLM testing | Revenue intelligence; testing sequence quality before it hits prospects | "Evals = A/B testing your sequences before they go live" |
| 12 | Observability, logging, tracing | Feedback loops; pipeline health monitoring and reply classification | "This tracing setup monitors your sequence performance in real time" |
| 13 | Deployment, CI/CD | Production GTM infrastructure; shipping and maintaining GTM systems | "This deploy pipeline ships your Clay tables and n8n workflows" |
| 14 | Cost optimization, latency | GTM stack cost management; optimizing API costs per enrichment run | "Every Clay credit is a token cost — optimize like you would LLM calls" |
| 15 | Security, auth, compliance | Outbound system security; webhook auth, API key rotation, CAN-SPAM/GDPR compliance, data handling for prospect records | "Your GTM stack has an attack surface — rotating API keys, securing webhooks, and handling prospect data under GDPR are the security layer most GTM engineers skip" |
| 16 | Distributed systems basics | Enrichment waterfall concurrency; parallel API calls across Clay providers with rate limits, retry logic, and partial failure handling | "Your enrichment waterfall is a distributed system — parallel requests, rate limit backpressure, idempotent retries. Same patterns, different domain." |
| 17 | MLOps, model lifecycle | GTM system lifecycle; versioning Clay tables (DVC), monitoring sequence performance drift, retraining scoring models when deal patterns shift | "MLOps for GTM = versioning your enrichment waterfalls, detecting when your scoring model drifts, and knowing when to retrain on fresh closed-won data" |
| 18 | Advanced prompting, chain-of-thought | Advanced personalization; multi-step research chains for ABM | "CoT prompting = how your agent reasons about an account before writing" |
| 19 | Retrieval-augmented generation | Knowledge-augmented outreach; using product docs and case studies in copy | "RAG = giving your outbound agent memory of your best customer stories" |
| 20 | AI systems design, capstone | Full-stack GTM system design; end-to-end pipeline from ICP to closed deal | "This is the capstone: design the full GTM system from scratch" |

---

## Priority Phases for GTM Redirect

Build these first — highest leverage for the Double Helix:

1. **Phase 11** — Evaluations → Revenue intelligence (evals are the feedback loop)
2. **Phase 14** — Cost optimization → GTM stack cost management (directly visible to practitioners)
3. **Phase 13** — Deployment → Production GTM infrastructure (ships the system)
4. **Phase 05** — LLM prompting → Outbound copy generation (highest student demand)
5. **Phases 15–17** — Security + distributed + MLOps → compliance, scale, lifecycle (the "grown-up GTM engineer" cluster)

---

## Context Dropped

The following were noted as out of scope for this integration doc:
- Specific lesson IDs (those live in the site repo, not here)
- Helix UI decisions (those belong in 00-d)
- Quiz format decisions (those belong in 00-a)
