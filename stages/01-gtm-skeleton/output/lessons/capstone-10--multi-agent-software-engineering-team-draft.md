# Capstone 10 — Multi-Agent Software Engineering Team

## Hook

Single agents hit a capability ceiling on complex software tasks — they lose coherence, forget requirements, and produce inconsistent code. Multi-agent teams divide cognitive load across specialized roles that critique and build on each other's work, the same way human engineering teams do.

## Concept

Three canonical multi-agent coordination patterns: **orchestrator-worker** (central coordinator dispatches tasks), **debate** (agents challenge each other's outputs), and **hierarchical** (architect delegates to implementers, who delegate to testers). Each pattern trades latency for quality differently. The mechanism that makes all three work is structured message passing — agents communicate through typed payloads, not free-form text, which keeps conversations from degrading.

## Mechanism

Implement message passing between agents using a shared state object and typed message schema. Each agent reads from an inbox, processes according to its role definition, and writes to outboxes. A router function moves messages between agents based on routing rules. The orchestrator maintains a task queue and termination condition (e.g., "all tests pass" or "reviewer approves"). Key failure modes: infinite loops between critiquing agents, context window overflow from accumulating message history, and role confusion when agents drift from their specialization.

## Use It

This is the same architecture behind multi-step GTM workflow automation — a research agent enriches a prospect, a copy agent drafts outreach, and a review agent checks brand compliance before sending. The orchestrator-worker pattern maps directly to the Clay waterfall: each agent is a enrichment/data step, the orchestrator is the waterfall controller, and the termination condition is "contact fully enriched or all sources exhausted." [CITATION NEEDED — concept: multi-agent GTM workflow orchestration in Clay]

## Ship It

**Exercise hooks:**

- **Easy:** Implement two agents — a coder and a reviewer — that pass messages through a shared state object. The coder writes a function, the reviewer returns "approve" or "reject with feedback," and the coder revises. Print the full message log. Run until approval or 3 revision cycles.

- **Medium:** Add a tester agent and an orchestrator. The orchestrator routes: spec → coder → reviewer → (if approved) → tester → (if tests pass) → done, else back to coder. All agents run locally using Claude Code subprocess calls. Output a final artifact (a Python module) and a trace of every message.

- **Hard:** Build a four-agent team (architect, coder, reviewer, tester) that takes a natural language feature description and produces a tested, reviewed Python module saved to disk. Implement retry limits, message history summarization after 10 messages, and a final quality report comparing multi-agent output to a single-agent baseline on the same prompt.

## Evaluate

Compare multi-agent output to single-agent output on identical tasks using three metrics: test pass rate, requirement coverage (did the output address every requirement in the spec), and revision cycles needed. Identify which coordination pattern produced the best results and where it broke down. Document failure modes observed: context overflow, role drift, infinite debate loops, or orchestrator routing errors. Every code example prints observable output — message logs, test results, and the final artifact — so verification is unambiguous.