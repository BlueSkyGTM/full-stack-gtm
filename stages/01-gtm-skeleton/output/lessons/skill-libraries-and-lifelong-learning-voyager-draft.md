# Skill Libraries and Lifelong Learning (Voyager)

## 1. Hook

You've built agents that call tools and follow plans. But every run starts from zero — re-learning what worked yesterday. Voyager solves this by persisting successful behaviors into a retrievable skill library, turning episodic agent runs into compounding capability.

## 2. Concept

Voyager operates three cooperating components: an automatic curriculum that proposes tasks, a skill library that stores and retrieves verified code snippets, and an iterative refinement loop that debugs failures before committing. The skill library is the memory mechanism — each skill is a self-contained function indexed by embedding search over its task description. Later tasks query the library, retrieve relevant prior skills, and compose them into new solutions rather than solving from scratch.

## 3. Mechanism

Explain the three-part architecture in order of data flow. First, the curriculum component reads the agent's current state (inventory, biome, nearby entities) and proposes a next task at appropriate difficulty. Second, the code generation component receives the task plus any retrieved skills and writes an executable program. Third, an execution environment runs the program and returns success, error, or state change. On success, the program is stored as a new skill. On failure, the error message feeds back into code generation for retry. Skills are embedded with sentence vectors and retrieved by cosine similarity to the current task description. The key insight: retrieval is semantic, not exact-match, so the library generalizes across situations.

## 4. Code

Build a minimal skill library system in Python. Implement skill storage with embedding search (using a simple hash-based embedding or a small model via API if available), skill retrieval given a task description, and a simulated execution loop that demonstrates adding skills and then composing them for a harder task. Print the retrieved skills and execution trace to show compounding. Exercises at easy (add a skill and retrieve it), medium (build the full store-retrieve-execute loop), hard (implement curriculum-driven task selection that queries the library before proposing).

## 5. Use It

In GTM, this maps to Zone 2 enrichment playbook libraries. Each successful enrichment waterfall (e.g., "resolve company from domain, then pull technographic data, then score fit") is a reusable skill. When a new ICP segment appears, you retrieve semantically similar playbooks rather than rebuilding from scratch. The compounding effect means your enrichment coverage grows with each shipped campaign. [CITATION NEEDED — concept: skill-library pattern applied to Clay waterfall reuse]

## 6. Ship It

Build a persistent skill library for a specific GTM workflow domain (enrichment sequences, outbound messaging templates, or qualification logic). Store at least 5 verified skills, implement semantic retrieval, and demonstrate that a new task retrieves and reuses prior skills. Output a log showing the retrieval hit, the composed solution, and the execution result. Hard mode: add the iterative refinement loop — if a skill fails on new data, capture the error, patch the skill, and re-store it.