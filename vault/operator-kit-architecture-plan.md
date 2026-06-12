<!-- /autoplan restore point: /c/Users/raymo/.gstack/projects/BlueSkyGTM-blueskygtm-engineering/master-autoplan-restore-20260612-033405.md -->
# Operator-Kit Architecture Plan: GLM Agent Integration

## Problem Statement

The 5 operator-kit agents (Lyra content, Lyra code, Echo, Hypatia, Newton) are intended to run on Z.ai GLM models. Claude Code only speaks the Anthropic Messages API format. Two viable approaches exist:

**Option A — Anthropic-to-GLM Proxy:**
Build a lightweight proxy server (FastAPI or Node/Express) that:
- Accepts requests in Anthropic Messages API format (`POST /v1/messages`)
- Translates and forwards to Z.ai GLM endpoint (`open.bigmodel.cn/api/paas/v4/`)
- Returns responses in Anthropic format (both streaming and non-streaming)
- Handles tool use translation (Anthropic `tool_use` blocks ↔ OpenAI `tool_calls`)
- Claude Code pointed at proxy via `ANTHROPIC_BASE_URL=http://localhost:3000`

**Option B — Standalone Python Scripts:**
Build operator-kit agents as standalone Python scripts using `zhipuai` SDK:
- Each agent is a Python script Claude Code shells out to via `Bash`
- Scripts call GLM directly — no format translation needed
- Claude Code orchestrates by reading stdout and passing context via args/stdin
- Cline can also call these scripts natively (GLM-native)

## Context

- `skills/operator-kit/` is currently empty (placeholder, `.gitkeep` only)
- `stages/00-c-agent-setup/output/` is empty — agent briefs not yet written
- Cline is in the stack and natively speaks GLM (OpenAI-compatible format)
- Z.ai API key confirmed working via `zhipuai` SDK
- `zhipuai` SDK installed at Python 3.14 site-packages
- `references/runtime-guide.md` defines agent routing — currently uses `<!-- Agent: Name -->` declarations and Claude Code Agent tool invocations
- Phase 0 runs entirely as Claude Code; GLM agents first active at Stage 01+
- 498 lessons to be generated (Lyra content is the volume workhorse)
- Model assignments: Lyra-content=GLM-5.1, Lyra-code=GLM-5, Echo=GLM-4.7-Flash, Hypatia=GLM-4.7, Newton=GLM-4.5-Air

## Constraints

- Must work within Claude Code's orchestration model
- Must be runnable from Cline as well
- Operator-kit needs to be built before Stage 01
- No breaking changes to existing `phases/` lesson structure
- `skills/operator-kit/` is tracked in git (never gitignored per CLAUDE.md)

## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Option B (standalone scripts) over Option A (proxy) | Taste | P5+P3 | Proxy SSE+tool_use translation = significant bug surface; Stage 01-04 agents only need read-brief+write-files | Option A (proxy) |
| 2 | CEO | MCP server deferred to TODOS.md pre-Stage 08 | Mechanical | P3 | Right long-term answer; premature for Stage 01; Stage 08 is the natural wiring stage | Option C now |
| 3 | CEO | Scope includes updating runtime-guide routing model | Mechanical | P1 | CONTEXT.md files still expect Agent tool invocation — must update dispatch declaration format | Leave as footnote |
| 4 | CEO | Use zhipuai SDK streaming, not raw stdout subprocess | Mechanical | P5 | Windows encoding + buffering risk on 498-lesson workload | Raw subprocess stdout |
| 5 | Eng | Pin zhipuai==2.1.5.20250725 in requirements.txt | Mechanical | P1 | SDK updates can silently change agent behavior; reproducible runs required | Unversioned |
| 6 | Eng | Add python-dotenv to requirements.txt | Mechanical | P1 | .env key loading needed across all agent scripts | Manual .env parse |
| 7 | Eng | Context loader fails gracefully if project-keywords.json missing | Mechanical | P5 | 00-c outputs don't exist yet at operator-kit build time | Silent KeyError |
| 8 | Eng | glm_client.py implements exponential backoff (3 retries) | Mechanical | P1 | 498 lessons = guaranteed rate limit hits; no retry = crashed pipeline | No retry |
| 9 | Eng | Each agent is a separate file, not if/elif monolith | Mechanical | P5 | Independent brief updates, easier debugging, no cross-agent coupling | Monolith |
| 10 | DX | Add .env.example to skills/operator-kit/ | Mechanical | P1 | TTHW blocked without it | Discover by reading code |
| 11 | DX | Add skills/operator-kit/README.md with invocation examples | Mechanical | P1 | runtime-guide describes intent; operator-kit needs concrete usage | Undocumented |
| 12 | DX | dispatch.py prints actionable error if zhipuai not installed | Mechanical | P5 | Raw ImportError is not actionable for operators | Raw ImportError |
