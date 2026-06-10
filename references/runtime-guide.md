# Runtime Guide: Full-Stack GTM Workspace

How the orchestration layer connects. Read this once at workspace setup. Stage contracts point here instead of re-explaining the machinery.

---

## Agent Routing

Claude Code reads the `<!-- Agent: -->` declaration at the top of each stage CONTEXT.md and invokes the matching operator-kit agent via the Agent tool with the CONTEXT.md as its brief.

| Declaration | Agent | Model | Primary job |
|---|---|---|---|
| `<!-- Agent: Lyra -->` | Lyra (content) | GLM 5.1 | Lesson drafts, exercise specs, quiz banks, outlines |
| `<!-- Agent: Lyra-code -->` | Lyra (code) | GLM 5.1 | Site components, Helix implementation, student state |
| `<!-- Agent: Newton -->` | Newton | GLM 5.1 | GTM research via Perplexity, source citations |
| `<!-- Agent: Echo -->` | Echo | GLM 5.1 | Codebase archaeology, read-only file traversal |
| `<!-- Agent: Hypatia -->` | Hypatia | GLM 5.1 | Curriculum audit, gap detection, quality challenge |

Agent brief files: `00-c/output/agent-briefs/`
Model config: `00-c/output/model-config.md`

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

gstack location: `~/.claude/skills/gstack/`

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

Install: `context-loader/install.md` in operator-kit repo.

---

## gbrain

gbrain persists structured knowledge across sessions. In this workspace it stores:

| What | Written by | Read by |
|---|---|---|
| Course identity + thesis | 00-e-seed | All Bu (Lyra) stages as standing context |
| GTM source citations | 00-b (Newton) | Stage 09 accuracy check |
| Agent brief summaries | 00-c | Stage 08 wiring |
| Audit findings | Stage 09 (Hypatia) | Stage 10 validation |

Run `/sync-gbrain` after each Phase 0 stage completes to keep cross-session context fresh.

---

