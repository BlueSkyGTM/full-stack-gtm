# Quiz factory

Batch workspace for Claude Code (or any agent) to finish curriculum quizzes **without chat mode**.

## Quick start

1. Regenerate the queue:

   ```bash
   python quiz-factory/scripts/generate_manifest.py --write
   ```

2. Open **`quiz-factory/`** as the working directory in Claude Code (or point the agent at this folder).

3. Agent reads **`CLAUDE.md`** → **[`lesson-planning`](../.cursor/skills/lesson-planning/SKILL.md)** → **`CONTEXT.md`** → **`REFERENCES.md`** → **`ARCHITECTURE.md`**.

4. Process `manifest.json` until the assigned phase slice is `done` or `blocked`.

5. PM spot-checks one quiz per phase before the next slice.

## Files

| File | Role |
|------|------|
| `CLAUDE.md` | Batch operator rules |
| `DESIGN.md` | Pointer to **lesson-planning** skill |
| `CONTEXT.md` | Frozen scope and phase order |
| `REFERENCES.md` | Schema, audit commands |
| `ARCHITECTURE.md` | Loops + claim-mapping pipeline |
| `QUALITY-RUBRIC.md` | PM spot-check per phase |

**Teaching / review:** [`.cursor/skills/lesson-planning/SKILL.md`](../.cursor/skills/lesson-planning/SKILL.md) — `/lesson-planning`
| `manifest.json` | Queue (`pending` / `done` / `blocked`) |
| `run.log` | Append-only run history (gitignored) |
| `templates/PHASE-BATCH.md` | Copy to `phases/NN-.../BATCH.md` per phase |

## Logs

```bash
# optional: append after each lesson
echo "2026-05-30T12:00:00Z | phases/07-.../01-foo | done | abc1234" >> quiz-factory/run.log
```

## After the factory

- Default CI: `python scripts/audit_lessons.py` (Tier A)
- Before fork cement: `python scripts/audit_lessons.py --strict-quiz` (Tier B)
- Push `main`; let CI rebuild `site/data.js`
