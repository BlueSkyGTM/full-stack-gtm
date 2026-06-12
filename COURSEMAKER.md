# The Coursemaker

## What This Is

This repository is two things. The first is a course: Full-Stack GTM Engineering, 502 lessons teaching AI engineering with GTM as the application layer. The second — and arguably the more valuable — is the system that produced it: a reproducible pipeline for anchoring any technical field to AI engineering and generating a structured, quality-assured, AI-tutored curriculum from the result.

The GTM content is one instantiation. The coursemaker is domain-agnostic.

---

## The Core Insight: Double Helix as Logic and Representation

Most course structures are flat. A topic list, a lesson order, a set of exercises. The content is organized, but it only has one access path: you navigate it linearly, or you search by keyword.

This pipeline produces a different structure — one with two access paths — because of a single design rule enforced at write time:

**Every AI engineering concept must have a GTM application mechanically derived from it. Not adjacent to it. Derived.**

This rule, called the Double Helix, does two things simultaneously:

**At the Logic Layer (Layer 1):** it is a curriculum constraint. Stage 02 of the build pipeline requires a chain-of-reason derivation step before any lesson prose is written. The agent must trace the specific mechanism — "because this AI concept works this way, it enables this specific GTM action" — before writing a single sentence. If the derivation doesn't hold, the lesson is flagged before it exists. This is what makes the GTM content accurate by construction rather than accurate by editorial review after the fact.

**At the Representation Layer (Layer 2):** it is an indexing structure. Because every GTM concept is semantically anchored to the AI concept that generates it, the content has dual access paths. You can retrieve by AI concept (what phase is this? what algorithm underlies it?) or by GTM use case (what does fine-tuning let me do in Clay? what does a vector store look like in a CRM?). The Sources block embedded in every lesson file — citations for every GTM claim, written at generation time — is not documentation. It is the index. No central database. No query-time lookup. The content is its own retrieval structure.

This is why Helix (the AI tutor) can answer GTM questions without a central knowledge base. The weave was created at write time. Helix inherits it.

---

## The Broader Application

Replace GTM with any field that has a practitioner knowledge base:

- Security engineering: AI concept → "where does this show up in AppSec?"
- Healthcare informatics: AI concept → "what does this enable in clinical decision support?"
- Legal technology: AI concept → "how does this apply to contract analysis or discovery?"
- Quantitative finance: AI concept → "what does this unlock in risk modeling?"

The coursemaker doesn't know about GTM. It knows about the Double Helix rule. Swap the application layer — swap the handbook, the redirect map, the citation sources — and the pipeline produces a different course with the same structural properties.

Every reference begets another one. The GTM handbook is a reference. Each lesson's Sources block cites from it. Each citation is a retrieval anchor for Helix. Each Helix interaction surfaces a new reference on demand from the antilibrary. The structure is generative: writing the course creates the knowledge graph that powers the tutor that teaches the course.

---

## The Pipeline (Stages 01–10)

The build pipeline converts source material into a complete, quality-audited, AI-tutored course:

| Stage | What It Does |
|-------|--------------|
| 00-a–e | Archaeology, topic mapping, agent briefing, Helix design, vault bootstrap |
| 01 | Generate 502 lesson outlines + manifest (the resumable queue) |
| 02 | Inject full hybrid lessons — chain-of-reason derivation, GTM weave, inline citations |
| 03 | Design CLI exercises that run entirely in Claude Code Desktop |
| 04 | Generate FSRS-formatted recall cards (format only; scheduling runs at Helix runtime) |
| 05 | Build Helix from the governed-maze spec; gate on 3-scenario test harness |
| 06 | Site readability overhaul + illustration generation (3-tier: GLM-image / Excalidraw / Mermaid) |
| 07 | Student state persistence |
| 08 | Verify all stage wiring; extend keyword routing |
| 09 | Quality pass: CLARITY, WEAVE, and ACCURACY judges on all 502 lessons; voice audit; gap fill |
| 10 | End-to-end validation — first-15-minutes onboarding through full student path |

The manifest pattern (borrowed from a proven 200-quiz factory) makes the pipeline resumable at any lesson. Stages never restart from zero. If Stage 02 stalls at lesson 247, it resumes at 247.

---

## Quality by Construction

Three mechanisms enforce quality before a human reviews anything:

**1. The derivation gate (Stage 02):** the GTM application must be derived from the AI concept before prose is written. A vague GTM claim has no AI anchor to derive from, so it can't pass the gate.

**2. The Sources block:** every GTM claim in every lesson requires an inline citation at write time. No citation = audit failure = lesson blocked in the manifest. Quality is a write-time constraint, not a post-hoc review.

**3. The judge layer (Stage 09):** three structured LLM-as-judge prompts (CLARITY, WEAVE, ACCURACY) run on all 502 lessons before the course ships. Each judge has a numeric pass threshold. Failed lessons are flagged as blocked in the manifest, revised, and re-evaluated — with a hard cap on revision loops to prevent infinite cycling.

---

## Helix: The Governed Maze

Helix is the AI tutor. It is not a conversational agent. It is not a forked persona. It is a governed maze: a structured decision tree where every student input routes to one of seven modalities (EXPLAIN, QUIZ, HINT, FLAG CHECK, CORRECT, ORIENT, REDIRECT), each with explicit trigger conditions and response rules.

The governing principle: Helix shows its reasoning. When it selects a modality, the reasoning chain is visible. This is the Open-Brain architecture — not because transparency is a feature, but because a tutor that improvises is a tutor that drifts. Drift, at 502 lessons, is a curriculum failure.

FSRS-5 spaced repetition handles recall scheduling in the background. Helix receives a card and a review summary (how many are due, which topics are overdue). The scheduling algorithm is invisible to Helix. The summary is not.

---

## The Third Structural Layer: Loop Engineering

*Added 2026-06-11. This is a living document — shifts are recorded here as they happen.*

Double Helix accounts for two layers: logic (the derivation constraint) and representation (dual-access indexing). What was missing — and was implicit in the build pipeline from the start — is the operational layer: how autonomous systems actually run, discover work, execute without constant human prompting, and maintain state across sessions.

That layer is **loop engineering**.

> "My job is to write loops." — Boris Cherny, Head of Claude Code, Anthropic

Loop engineering is replacing yourself as the person who prompts the agent. You design the system that does it instead. The five primitives: scheduling (cadenced discovery), worktrees (parallel isolation), skills (persistent knowledge), connectors (MCP), and sub-agents (maker/checker split) — plus memory/state as the durable spine.

**Why this changes the coursemaker:**

The build pipeline was already a loop. batch-orchestration is the worktree primitive. CONTEXT.md files are skills. vault/ is state. autoplan is the maker/checker pattern. But without the explicit framework, the standards were informal. At 498-lesson scale, informal breaks. LOOP.md (at repo root) now codifies the operating model before any Stage runs.

**Why this changes the curriculum:**

The GTM operator kit — TAM segmentation, signal detection, Clay enrichment, HubSpot qualification — is a loop. Every GTM system a student will build after this course is a loop. Teaching GTM engineering without teaching loop design is training technicians, not engineers.

The positioning is not "future-proofing." It is engineering from first principles. The Boris Cherny quote is the opening frame: the leverage point has moved from prompting to loop design. A GTM engineer who cannot design autonomous signal-to-outreach loops will be replaced by one who can. This course closes that gap.

Loop engineering threads through the curriculum as the operating model — introduced early as a frame (the Boris Cherny quote, the five primitives, why the leverage point shifted), deepened in the GTM execution phases as the architecture underlying every system students build, and synthesized in the capstone as the student's own loop design. It does not compete with the AI engineering depth. It explains why the AI engineering depth exists: you need to understand what's under the loop to build one that doesn't break.

**On GLM air replacing Perplexity:**

Newton's gap-fill research role (activated when curriculum gaps are detected) runs on GLM air, not Perplexity. GLM air's speed and tooling affinity make it a better fit for the maker/checker loop pattern than a search-first API. This is a build pipeline decision that propagates into the curriculum: students learn GLM air as the research agent in their own loops, not a standalone search tool.

---

## References

This coursemaker drew from seven external sources, each contributing a specific structural layer:

| Source | What It Contributed |
|--------|-------------------|
| **Made With ML** | The 498-lesson AI engineering curriculum — the spine of the Double Helix |
| **80/20 GTM Engineering Playbook** | The application layer — toolstacks, KPIs, workflows, copy templates sourced from six-figure GTM practitioners |
| **Open-Brain / governed-maze pattern** | The reasoning chain architecture Helix is built against — structured decision flow with explicit nodes. Professor Synapse was evaluated and discarded (blackbox persona, general-purpose tutoring, hidden constraints). Helix is built from scratch against vault/helix-architecture.md. |
| **learning-commons-org/evaluators** | LLM-as-judge architecture; CLEAR text complexity framework adapted into CLARITY_JUDGE |
| **EducationQ** | Teacher→student→evaluator loop pattern adapted into the Helix test harness |
| **axtonliu/smart-illustrator + coleam00/excalidraw-diagram-skill** | 3-tier illustration pipeline; render-and-inspect self-validation loop |
| **cobusgreyling/loop-engineering** | The operational layer — five primitives, phased rollout (L1/L2/L3), maker/checker standard, STATE.md pattern, anti-patterns catalog. Integrated into LOOP.md and threaded through the curriculum as the GTM loop engineering arc. |

None of these were taken wholesale. Each was a reference that became a component — adapted, constrained, and integrated into a pipeline that none of them individually describe.

---

## Why a 3-Month Programmer Can Build This

The coursemaker is not a software project in the traditional sense. No production server. No user authentication. No database schema migrations. The file system is the architecture. Markdown files are the contracts. The AI agents are the workers.

What makes it possible is the same thing that makes the Double Helix work: structure enforced at write time. Every stage has a CONTEXT.md that tells the agent exactly what to read, what to produce, and what constitutes a passing audit. The human writes the spec. The agent executes it. The quality layer catches what the agent misses.

The loop engineering layer adds the third rail: the agents aren't just executing specs, they're running loops — with state, with verification, with human gates at the right moments. The coursemaker is itself an example of what it teaches. Building it is the proof of concept for the operational model the students will learn.

The skill being developed is not "how to code a course platform." It is how to design systems where AI agents can do reliable, auditable work at scale — and how to build the quality gates that make that reliability verifiable.

That is AI engineering. That is loop engineering. The GTM course is just the first instantiation.
