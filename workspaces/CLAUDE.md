# Claude Code — Dean · Horizon Coder

**Mission: Claude Code designs and builds. Cline executes schema repair and batch work.**

This is the project root. Four workspaces live below.
Claude Code operates from here — not from inside any workspace.

---

## What this project is

A personal AI engineering curriculum and portfolio site. 498 lessons, 20 phases, ~320 hours
across Python, TypeScript, Rust, and Julia. The curriculum content is complete.
The site is built and deployed. What remains is schema normalization and WordPress deployment.

---

## Agent roles

| Agent | Role | What it owns |
|-------|------|--------------|
| **Claude Code (you)** | Dean · Horizon Coder | Architecture decisions, batch briefs, site work, WordPress deployment |
| **Cline** | Inline Coder | Quiz schema repair execution, commit work |

**Routing rule:** Cline executes schema repair batches from briefs. Claude Code handles
everything that requires judgment — site changes, deployment, new content.

---

## The workspaces

| Workspace | Directory | Role |
|-----------|-----------|------|
| **AI School** | `ai-school-curriculum/` | Curriculum + portfolio site (`site-new/`) |
| **Expansion** | `ai-school-expansion/` | Made With ML source — future content absorption |
| **Anti-library** | `ai-school-anti-library/` | Free books source (EbookFoundation) |
| **Workspace Builder** | `workspace-builder/` | Automated setup — read only |

`ai-school-website/` was deleted. All site work is in `ai-school-curriculum/site-new/`.

### Also at root
- `cline-backlog/` — batch briefs for Cline, ACTIVE.md
- `CONTEXT.md` — current work queue and known state

---

## Curriculum state

- **498 lessons** across 20 phases (30 new Phase 19 lessons added 2026-06-03)
- **1,109 quiz questions** across 200+ files — pulled from upstream 2026-06-03
- **Quiz creation: DONE.** Do NOT brief Cline to create quizzes. The data exists.
- **Schema repair: IN PROGRESS.** 162 files still need normalization to 6q format.

### Quiz schema — canonical shape, no exceptions

```json
{
  "lesson": "<dir-slug>",
  "title": "<Lesson Title>",
  "questions": [
    {"stage": "pre",   "question": "", "options": ["","","",""], "correct": 0, "explanation": ""},
    {"stage": "check", "question": "", "options": ["","","",""], "correct": 1, "explanation": ""},
    {"stage": "check", "question": "", "options": ["","","",""], "correct": 2, "explanation": ""},
    {"stage": "check", "question": "", "options": ["","","",""], "correct": 1, "explanation": ""},
    {"stage": "post",  "question": "", "options": ["","","",""], "correct": 3, "explanation": ""},
    {"stage": "post",  "question": "", "options": ["","","",""], "correct": 0, "explanation": ""}
  ]
}
```

---

## Work queue

| Priority | Job | Scope | Agent |
|----------|-----|-------|-------|
| **P0** | schema_repair | 162 files across 10 phases (see CONTEXT.md) | Cline |
| **P1** | WordPress deployment | Install plugin + page template on WP install | Claude Code |
| **P2** | Quiz UI | Build lesson reader quiz integration | Claude Code (future) |
| **P3** | Graphify | Interactive prerequisite DAG for site | Claude Code (future) |

**DO NOT create quizzes.** Data exists. Schema repair only.

---

## Portfolio site (site-new/)

Six-page vanilla JS/CSS dashboard at `ai-school-curriculum/site-new/`.
WordPress plugin and page template at `site-new/wordpress/`.
GitHub: https://github.com/BlueSkyGTM/ai-school

Pages: Home · Course · Catalog · Library · Projects · Glossary

---

## Commit rules

- One commit per lesson/fix. Never batch commits.
- Format: `fix(phase-NN/MM): repair quiz schema` or `feat(phase-NN/MM): add <slug> lesson`
- Subject ≤72 chars.

## CI gate

```bash
python3 scripts/audit_lessons.py phases/<slug>/
python3 scripts/audit_lessons.py --strict-quiz phases/<slug>/
```

---

## gstack skills

| Skill | When |
|-------|------|
| `/qa` | Before merging site changes |
| `/design-review` | Visual polish on site |
| `/investigate` | When root cause is unclear |
| `/review` | Before any PR |
| `/ship` | When ready to release |
| `/browse` | All web browsing — always use this |

**Web browsing:** always use `/browse`. Never use `mcp__claude-in-chrome__*` tools.

---

## Hard rules

- **Never brief Cline to create quiz.json files** — they already exist
- Never commit multiple lessons in one commit
- Never edit `site/data.js` (old site) — unused
- Always verify stage sequence: `pre, check, check, check, post, post`
- One workspace per session — never mix concerns
