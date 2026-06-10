---
name: refresh
version: 1.0.0
description: >-
  Rebuild or verify the local code knowledge graph after curriculum or site
  edits. Use with /refresh, "refresh the graph", "is the graph stale", or
  "graph refresh". Does not install tools or Cursor rules.
disable-model-invocation: true
---

# Refresh

User-facing name for **graph refresh** — rebuild the local AST code graph.

## When

Run after edits to lesson `code/`, `docs/`, `outputs/`, or `site/`.

Skip after README, ROADMAP, glossary, or `quiz.json`-only changes.

## Commands (use these — do not suggest tool install or cursor rule install)

**Rebuild:**

```bash
python3 scripts/refresh_graph.py
```

**Check freshness vs git HEAD:**

```bash
python3 scripts/verify_graph.py --always
python3 scripts/verify_graph.py --strict    # implementer gate; must exit 0
```

**Explore before grep (read-only):**

```bash
python3 scripts/query_graph.py query "<narrow question>"
python3 scripts/query_graph.py path "<A>" "<B>"
python3 scripts/query_graph.py explain "<concept>"
```

## Workflow

| Who | Action |
|-----|--------|
| User says **/refresh** | Run `refresh_graph.py`, report exit code |
| User asks if graph is stale | Run `verify_graph.py --always` |
| Cline after code/site work | `refresh_graph.py` then `verify_graph.py --strict` |
| Otto exploring structure | `query_graph.py` only — never rebuild |

## Do not

- Mention or run package **install** commands for Cursor rules
- Do not read the gitignored local graph output directory into context
- Commit the gitignored local graph output directory

Canonical policy: `.claude/rules/local-graph.md`.
