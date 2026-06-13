# Repo Memory and Durable State

## Hook
Your AI coding assistant starts every session with total amnesia. Two hours of pair programming gone the instant you close the terminal. Repo memory is the mechanism that fixes this — structured files loaded automatically into context so the assistant carries forward decisions, conventions, and architecture without you re-explaining everything.

## Concept
Define the three layers of persistent state available inside a repo: project-level memory files (like `CLAUDE.md`) loaded on every invocation, session-scoped artifacts that accumulate during a conversation, and external durable stores (SQLite, JSON files, git-tracked configs) that survive across sessions and machines. Explain the read-order precedence: when multiple memory sources exist, which one wins and why. Cover the tradeoff between context window consumption and recall fidelity — every line of memory you load costs tokens.

## Demo
Build a working repo memory system from scratch. Create a `CLAUDE.md` file with project conventions, then demonstrate Claude Code loading and applying those conventions in a real task. Then write a Python script that writes and reads a durable JSON state file — tracking a mock "enrichment run" with timestamps and status — and prints the state transitions to stdout so the mechanism is observable.

## Use It
This concept maps directly to GTM enrichment pipeline state management. In Clay, a waterfall enrichment that runs across multiple rows needs to track which rows succeeded, which failed, and which need retry — that's durable state. The same read-order precedence problem appears when you have multiple Clay tables feeding a single campaign: which table's status column is the source of truth? [CITATION NEEDED — concept: Clay waterfall state management and retry logic]

**Exercise hooks:**
- (Easy) Create a `CLAUDE.md` file that encodes three project conventions, then verify Claude Code applies them.
- (Medium) Write a Python script that implements a simple state machine (pending → running → done/failed) persisted to a JSON file, with idempotent restart capability.
- (Hard) Build a durable state tracker that logs every AI code generation session's inputs and outputs to a SQLite database, with a query that surfaces "what did I change last week."

## Ship It
Implement a repo memory stack for your own project: a `CLAUDE.md` with architecture decisions, a `.claude/` directory for session artifacts, and a durable state file that tracks what the AI assistant has done across sessions. The GTM connection: this is identical to maintaining enrichment run logs and deduplication state in a GTM data pipeline. You wouldn't re-enrich 10,000 contacts every Monday — you'd checkpoint state and resume. Same mechanism, different domain. [CITATION NEEDED — concept: GTM pipeline checkpoint and resume patterns]

## Evaluate
Quiz questions grounded in observable behavior: what file does Claude Code load first when multiple memory sources exist, what happens to context window budget as memory files grow, and how to detect stale durable state that no longer matches the repo's current structure. All questions must be answerable from the demo output or direct experimentation.