# Current State — AI Engineering School
**Last updated:** 2026-06-03

---

## What is done

### Curriculum
- 498 lessons across 20 phases (30 new Phase 19 capstone lessons added today)
- All lesson content exists — no missing lessons to create
- 1,109 quiz questions across 200+ files pulled from upstream
- Quiz creation is COMPLETE — do not write quiz creation batches

### Portfolio site (`site-new/`)
- All 6 pages built and working: Home, Course, Catalog, Library, Projects, Glossary
- Progress tracking via localStorage (adapter-ready for WordPress REST)
- 110 library resources with type filter (Book/Video/Course/Paper/Article/Interactive)
- XP/level/rank system (7 tiers: Initiate → AI Architect)
- All 60 capstone projects surfaced on Projects page
- Mobile nav fixed, command palette working, no JS errors
- WordPress plugin + page template written at `site-new/wordpress/`
- README written and pushed to https://github.com/BlueSkyGTM/ai-school

---

## What still needs doing

### P0 — Schema repair (Cline executes from batches)

162 quiz.json files need normalisation to the 6-question canonical schema.
Batches 001-010 in `cline-backlog/batches/pending/` cover phases 05 and 14.
New batches needed for the remaining phases.

| Phase | Files | Issue | Status |
|-------|-------|-------|--------|
| 00-setup-and-tooling | 12 | 5q | No batch yet |
| 01-math-foundations | 22 | 5q | No batch yet |
| 04-computer-vision | 28 | 5q | No batch yet |
| 05-nlp | 29 | 8q → 6q | Batches 001-005 written |
| 07-transformers | 1 | 5q | No batch yet |
| 11-llm-engineering | 4 | 5q | No batch yet |
| 14-agent-engineering | 42 | 7q → 6q | Batches 006-010 written |
| 16-multi-agent | 2 | 5q | No batch yet |
| 17-infrastructure | 1 | 7q → 6q | No batch yet |
| 19-capstone-projects | 21 | 7q → 6q | No batch yet |

**Active batch:** `pending/batch-006-phase14-01-09.md`

For 5q files: audit each to determine if they need 1 question added or different handling.
For 7-8q files: trim per existing batch pattern.

### P1 — WordPress deployment

Plugin and template are written. Need to install on a WordPress instance.
Files: `ai-school-curriculum/site-new/wordpress/`
- `aischool-progress.php` — REST plugin (`/wp-json/aischool/v1/progress`)
- `page-aischool.php` — page template for Course/Catalog/Library/Projects/Glossary

### P2 — Quiz UI (future)

1,109 questions exist as JSON. Nothing renders them yet. Build a quiz interface
into the lesson reader when WordPress is deployed.

### P3 — Graphify (future)

Interactive prerequisite DAG for the site. Shows chapter dependencies visually.

---

## Known gaps (do not skip)

1. **data.js lesson count** — says "470+" in some copy but site now has 498 lessons.
   Update signpost text on index.html when convenient.

2. **Stage sequence not enforced by CI** — manually verify `pre, check, check, check, post, post`
   before every schema repair commit until the audit script is updated.

---

## How to brief Cline

1. Write `cline-backlog/batches/pending/batch-NNN-description.md`
2. Set `cline-backlog/batches/ACTIVE.md` → relative path to that file
3. Hand off to Cline: "read ACTIVE.md and execute"
4. Cline commits per lesson, you review

Every brief needs a locked JSON skeleton. Cline never invents structure.

Gold reference quizzes (style anchors — never copy content):
- `phases/07-transformers-deep-dive/16-speculative-decoding/quiz.json`
- `phases/10-llms-from-scratch/34-gradient-checkpointing/quiz.json`
- `phases/19-capstone-projects/54-paper-writer/quiz.json`
