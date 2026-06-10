---
description: Delegate bulk and parallel work to subagents before doing it in main context (Cline)
alwaysApply: true
---

# Subagent-first (Cline)

Before file edits, search, or verification in main context, ask: **can a subagent do this?**

**Use subagents for:** bulk file ops, parallelizable checks, bounded exploration, reading many files.

**Do yourself for:** sign-off, commits/push, single-line edits, final reports, whole-repo coordination.

**Do not use subagents for:** git operations, one-line table rows, compiling cross-agent summaries.

**For batch quiz work specifically:** use the `/batch-orchestration` skill (`.claude/skills/batch-orchestration/SKILL.md`). It defines the manifest-queue pattern, BATCH.md knowledge injection, and round protocol for all `create_quiz` / `schema_repair` / `fill_explanations` / `redo_quiz` jobs.

See `CURRICULUM_QA_FINDINGS_2026-05-30.md` for a worked example (2026-05-30 sprint).
