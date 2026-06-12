# Primitives Matrix — Tool Comparison

How the five primitives map across the three major agentic coding tools.

| Primitive | Grok | Claude Code | Codex |
|-----------|------|-------------|-------|
| **Scheduling** | `/loop` (native interval), `/goal` (condition-based) | `CronCreate` / `ScheduleWakeup` (harness), manual trigger per stage | Codex tasks, GitHub Actions dispatch |
| **Worktrees** | `isolation: "worktree"` on Agent spawn | `EnterWorktree` tool; `isolation: "worktree"` on Agent spawn | Separate sandboxes per run |
| **Skills** | SKILL.md + skill plugins registered in system prompt | SKILL.md + CONTEXT.md files; `/` slash command invocation | `AGENTS.md` + custom instructions |
| **Connectors (MCP)** | MCP plugins, built-in Linear/GitHub/Slack tools | MCP servers in `~/.claude.json`; gstack wraps common ones | MCP support experimental as of 2026 |
| **Sub-agents (maker/checker)** | `Agent()` tool call with `subagent_type`; `isolation: "worktree"` | `Agent()` tool call with `subagent_type`; worktree isolation | Parallel task runs; no native verifier pattern |
| **Memory / State** | STATE.md in repo; Linear board sections | `vault/` + manifest + `pipeline_runs`; auto-memory in `.claude/` | Per-run context only; no durable cross-run state |

## Notes

**Grok** has the most mature native loop primitives — `/loop` and `/goal` are purpose-built for this. Worktree isolation is first-class. Checker split requires explicit wiring but the pattern is well-supported.

**Claude Code** (this project's primary tool) achieves equivalent patterns through a combination of harness tools (`CronCreate`, `ScheduleWakeup`, `EnterWorktree`, `Agent`) and explicit skill files. The batch-orchestration skill is our worktree/parallel-subagent wrapper. MCP connectivity is mature. The gstack skill suite is our connectors + skill layer.

**Codex** is best for isolated, single-session runs. Loop patterns are possible via GitHub Actions but require more external scaffolding. Use Codex for bounded tasks that don't need cross-run state.

## In This Project

| Primitive | How We Implement It |
|-----------|-------------------|
| Scheduling | Manual trigger per stage (L1 by design — human-gated) |
| Worktrees | `batch-orchestration` skill, up to 8 parallel subagents |
| Skills | `CONTEXT.md` files + `vault/` shared layer + gstack `/` commands |
| Connectors | gstack suite + GLM air (research) + MCP for GTM pipeline |
| Sub-agents | Lyra (maker), Hypatia (checker), autoplan (plan checker) |
| Memory / State | `vault/` + manifest + `pipeline_runs` audit log |
