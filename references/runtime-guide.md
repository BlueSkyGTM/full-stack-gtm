# Runtime Guide: Full-Stack GTM Workspace

How the orchestration layer connects. Read this once at workspace setup. Stage contracts point here instead of re-explaining the machinery.

---

## Skill Routing

Operator-kit skills are gstack **agentic skills** — each `SKILL.md` drives a full behavior chain: read context → call Z.ai GLM → write output → validate. No proxy, no Railway, no backing Python scripts. The skill is the harness; GLM is the worker. Phase 0 runs entirely as Claude Code; skills first deploy at Stage 01+.

**Invocation:** Claude Code calls these as slash commands (e.g. `/write-lesson`), not via the Agent tool. Each `skills/operator-kit/<skill>/SKILL.md` contains the full chain including embedded bash that calls Z.ai directly.

**Endpoint:** `https://api.z.ai/api/coding/paas/v4` — simple `Bearer $ZHIPUAI_API_KEY` auth. No JWT signing. Set `ZHIPUAI_API_KEY` and `ZAI_BASE_URL` in `.env` (gitignored). See `vault/operator-kit-architecture-plan.md` for full setup and confirmed model list.

| Skill | Model | Primary job | First active |
|---|---|---|---|
| Claude Code directly (no skill) | — | All Phase 0 stages | Stages 00-a through 00-f |
| `/write-lesson` | GLM-5.1 | Lesson drafts and outlines | Stage 01 |
| `/write-exercise` | GLM-5.1 | Exercise specs | Stage 03 |
| `/write-quiz` | GLM-5.1 | Quiz banks (FSRS-ready) | Stage 04 |
| `/build-site-component` | GLM-5 | Site components, Helix implementation, student state | Stage 05 |
| `/scan-repo` | GLM-5-Turbo | Read-only codebase traversal, archaeology | Stage 01+ |
| `/quality-check` | GLM-5-Turbo | Curriculum audit, gap detection, accuracy challenge | Stage 09 |
| `/find-citations` | GLM-5-Turbo | Gap-fill research, citation finding | Stage 01+ |

**`/find-citations`:** activates when `/quality-check` or any stage flags curriculum gaps (missing source material, undercited concepts, topics the handbook doesn't cover). Uses `gtm-integration-citations.md` as the link repo. Fills blanks without stopping the main build batch. Brief written in 00-c for immediate deployment at Stage 01+.

Skill brief files (written by 00-c): `stages/00-c-agent-setup/output/agent-briefs/`
Model config: `stages/00-c-agent-setup/output/model-config.md`

---

## gstack Skill Triggers

Skills fire at specific contract moments. They are not ambient — each stage contract names exactly which skills it triggers and when.

| Moment | Skill | Who triggers |
|---|---|---|
| Stage entry (before anything runs) | `/spec` | Claude Code |
| Before any GLM code merges | `/review` | Claude Code |
| Stage close / release | `/ship` | Claude Code |
| Any visual site change | `/design-review` | Claude Code (Stage 06 entry) |
| Direction pivot or scope change | `/office-hours` | Claude Code |
| Post-deploy monitoring | `/canary` | Claude Code |
| Full plan review gate | `/autoplan` | User-triggered |

**gstack is not a terminal CLI.** All skills are slash commands invoked inside Claude Code. There is no `gstack` shell command to type — use `/office-hours`, `/review`, `/qa`, `/spec`, etc. directly in the Claude Code prompt.

**Locations:**
- Global install (active): `~/.claude/skills/gstack/` — built and registered via `./setup`
- Repo copy (tracked in git): `skills/gstack/` — source of truth, never gitignored
- To rebuild after pulling changes: `cd ~/.claude/skills/gstack && ./setup`

**Full skill list (all available as slash commands):**

| Category | Skills |
|---|---|
| Planning | `/spec`, `/autoplan`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/plan-devex-review`, `/plan-tune`, `/office-hours` |
| Review & QA | `/review`, `/qa`, `/qa-only`, `/design-review`, `/devex-review`, `/cso`, `/health` |
| Design | `/design-consultation`, `/design-html`, `/design-shotgun` |
| Ship | `/ship`, `/land-and-deploy`, `/canary`, `/document-release` |
| Browser | `/browse`, `/connect-chrome`, `/setup-browser-cookies`, `/scrape` |
| Docs & Learn | `/document-generate`, `/learn`, `/retro`, `/investigate`, `/make-pdf` |
| Memory | `/setup-gbrain`, `/sync-gbrain`, `/context-save`, `/context-restore` |
| Maintenance | `/gstack-upgrade`, `/freeze`, `/unfreeze`, `/guard`, `/careful`, `/skillify` |

---

## Context Loader

The context loader (operator-kit) auto-injects project files into context when a stage keyword appears in a prompt. Configured in `00-c/output/project-keywords.json`.

| Keyword | Files injected |
|---|---|
| `gtm-skeleton` or `stage-01` | 00-a format specs, 00-b topic map, 00-e-seed variable registry |
| `lesson-injection` or `stage-02` | Stage 01 outlines, hybrid-lessons path |
| `exercise-design` or `stage-03` | Stage 02 lessons, copy-paste flag format |
| `quiz-recall` or `stage-04` | Stage 02 lessons, Stage 03 exercises, FSRS spec |
| `helix-build` or `stage-05` | 00-d design spec, Stage 03-04 outputs |
| `site-readability` or `stage-06` | 00-a design snapshot, existing render pipeline |
| `student-state` or `stage-07` | 00-a auth audit, 00-d state options, Stage 05 Helix model |
| `agent-wiring` or `stage-08` | 00-c agent briefs, project-keywords.json |
| `quality-pass` or `stage-09` | Stage 02 lessons, source citations, Helix voice |
| `validation-run` or `stage-10` | Full curriculum, live site |

Context loader config is written by 00-c to `00-c/output/project-keywords.json`. No external install required.

---

## Tooling Layer — Three Distinct Tools

These are not the same thing. Do not conflate them.

### graphify — for subagents

Builds the local code graph so operator-kit subagents (Echo, Lyra-code, Newton, Hypatia) can traverse the codebase semantically rather than brute-force grep.

Scripts (written in 00-f): `scripts/query_graph.py`, `scripts/verify_graph.py`, `scripts/refresh_graph.py`
Output: `graphify-out/` (gitignored — generated, not committed)
Consumer: every operator-kit subagent that needs to navigate code

### gbrain — for the host + gstack skills

Persists cross-session knowledge for the host (Claude Code) and gstack skills. Stores CEO plans, decisions, source citations, agent brief summaries, audit findings.

| What | Written by | Read by |
|---|---|---|
| Course identity + thesis | 00-e-seed | All Lyra stages as standing context |
| GTM source citations | 00-b (Newton) | Stage 09 accuracy check |
| Agent brief summaries | 00-c | Stage 08 wiring |
| Audit findings | Stage 09 (Hypatia) | Stage 10 validation |

Setup: `/setup-gbrain` — configured in 00-f after project structure is known
Sync: `/sync-gbrain` after each Phase 0 stage completes

### Helix open brain — for students

gbrain + FSRS as the student memory layer. Tracks what a learner knows and when to resurface it. This is a product feature, not infrastructure — built in Stage 05 as part of the Helix agent.

Not the same as gbrain. Helix open brain is student-facing; gbrain is builder-facing.

---

