---
description: local code graph — query-first, refresh via /refresh skill, verify gate
alwaysApply: true
---

Local code graph output is gitignored under `graphify-out/`. One-time package install: see `/refresh` skill.

Do not read `.claude/_archive/` unless the user asks to restore. **Never run cursor rule install** for the graph tool.

## Query-first (Otto — read-only)

Before broad grep or file spelunking, if `graphify-out/graph.json` exists:

```bash
python3 scripts/query_graph.py query "<narrow question>"
python3 scripts/query_graph.py path "<A>" "<B>"
python3 scripts/query_graph.py explain "<concept>"
```

- Never load `graph.json` into context.
- `GRAPH_REPORT.md`: Corpus/Summary, Graph Freshness, God Nodes, Surprising, Suggested Questions only — skip Community Hubs and Communities.
- May run `python3 scripts/verify_graph.py` (no `--strict`). Never run `scripts/refresh_graph.py`.

## Refresh (user + Cline)

**`/refresh`** → `.claude/skills/refresh/SKILL.md`.

After lesson `code/`, `docs/`, `outputs/`, or `site/` edits:

```bash
python3 scripts/refresh_graph.py
```

Skip when only README, ROADMAP, glossary, or `phases/**/quiz.json` changed.

## Verify gate (Cline)

After refresh: `python3 scripts/verify_graph.py --strict` must exit 0.

## Do not

Commit local graph output, add CI/hooks for graph maintenance, or duplicate this rule elsewhere.
