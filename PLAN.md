# Plan: Full-Stack GTM Repo Restructure

## Goal

Align the `full-stack-gtm` repo with the ICM scaffold generated in Stage 03 of the workspace-builder pipeline.
The repo should function as a Full-Stack GTM ICM workspace: CLAUDE.md routes Claude Code into the right stage, Graphify is gone, and no duplicate content exists.

## Current State

```
full-stack-gtm/ (root)
├── CLAUDE.md                  ← WRONG: ICM root CLAUDE.md (came in via merge), not FSG workspace CLAUDE.md
├── phases/                    ← KEEP: actual course lesson content (498 lessons)
├── assets/                    ← KEEP: course assets
├── library-catalog/           ← KEEP: lesson catalog
│   └── .claude/skills/graphify/ ← DELETE: Graphify skill
├── scripts/                   ← PARTIAL: keep non-graphify scripts, delete graphify ones
│   ├── check_graphify_freshness.py  ← DELETE
│   ├── fix_graphify_labels.py       ← DELETE
│   ├── query_graph.py               ← DELETE (graphify graph query)
│   ├── refresh_graph.py             ← DELETE (graphify graph refresh)
│   ├── verify_graph.py              ← DELETE (graphify graph verify)
│   └── [others: audit_lessons, build_catalog, scaffold, etc.] ← KEEP
├── .graphifyignore            ← DELETE
├── workspaces/full-stack-gtm/ ← DELETE: duplicate snapshot of old course content nested as subfolder
├── [ICM scaffold files]       ← MISSING: CONTEXT.md, stages/, vault/, references/, shared/, skills/
└── [course root files]        ← KEEP: phases/, assets/, quiz-factory/, professor-synapse/, etc.
```

## Target State

```
full-stack-gtm/ (root)
├── CLAUDE.md          ← REPLACE: with Stage 03 scaffold CLAUDE.md (Full-Stack GTM workspace routing)
├── CONTEXT.md         ← ADD: from Stage 03 scaffold
├── setup/             ← ADD: from Stage 03 scaffold (questionnaire.md)
├── references/
│   └── runtime-guide.md  ← ADD: from Stage 03 scaffold
├── vault/             ← ADD: from Stage 03 scaffold (variable-registry, course-identity-doc, etc.)
├── shared/            ← ADD: from Stage 03 scaffold (gtm-handbook-extract.md)
├── skills/            ← ADD: from Stage 03 scaffold (gstack/ + operator-kit/ placeholders)
├── stages/            ← ADD: from Stage 03 scaffold (16 stage directories with CONTEXT.md stubs)
├── phases/            ← KEEP (existing course content)
├── assets/            ← KEEP
├── library-catalog/   ← KEEP (minus graphify skill)
├── scripts/           ← KEEP (minus 5 graphify scripts)
├── quiz-factory/      ← KEEP
├── professor-synapse/ ← KEEP
└── [other course files] ← KEEP
```

## Work Items

### W1: Replace root CLAUDE.md
- Source: `workspaces/workspace-builder/stages/03-scaffolding/output/full-stack-gtm/CLAUDE.md`
- Action: overwrite root `CLAUDE.md`

### W2: Add ICM scaffold files
Copy from `workspaces/workspace-builder/stages/03-scaffolding/output/full-stack-gtm/` to root:
- `CONTEXT.md`
- `setup/questionnaire.md`
- `references/runtime-guide.md`
- `vault/` (all 4 files)
- `shared/gtm-handbook-extract.md`
- `skills/.gitkeep`
- `stages/` (all 16 stage directories with their CONTEXT.md files)

### W3: Purge Graphify (root)
Delete:
- `.graphifyignore`
- `scripts/check_graphify_freshness.py`
- `scripts/fix_graphify_labels.py`
- `scripts/query_graph.py`
- `scripts/refresh_graph.py`
- `scripts/verify_graph.py`
- `library-catalog/.claude/skills/graphify/`

### W4: Remove nested workspaces/full-stack-gtm/ subfolder
The snapshot commit nested the old repo as `workspaces/full-stack-gtm/`. This duplicates root-level content.
Delete the entire `workspaces/full-stack-gtm/` subfolder.

### W5: Purge Graphify (workspaces/full-stack-gtm/ — only if W4 is deferred)
Only relevant if W4 is not executed.

## Key Decision

**D1: What to do with `workspaces/` at root?**
The root currently has a `workspaces/` folder containing `full-stack-gtm/` (old nested content) plus other loose items.
Option A: Delete `workspaces/full-stack-gtm/` but keep other `workspaces/` subfolders
Option B: Delete the entire `workspaces/` folder (all subfolders are legacy)
Option C: Keep `workspaces/` folder, just clean `workspaces/full-stack-gtm/`

**D2: Do ICM `stages/` and `phases/` (course content) coexist at root?**
ICM pipeline stages (00-a through 10) are agent workflow scaffolding.
Course content phases (phases/01-intro/, etc.) are actual lesson files.
They serve different purposes and won't collide on names.
Coexistence is correct — `stages/` is the build pipeline, `phases/` is the output.

## Out of Scope
- Editing lesson content in `phases/`
- Touching `professor-synapse/`
- Any changes to vercel.json, .github/, api/

## Dependencies
- Stage 03 scaffold output at: `../Interpreted-Context-Methdology/workspaces/workspace-builder/stages/03-scaffolding/output/full-stack-gtm/`
