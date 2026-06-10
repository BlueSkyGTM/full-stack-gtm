---
name: housekeeping
version: 1.0.0
description: >-
  Audit repo for context, folder, and code bloat; consolidate and remove
  contradictions. Always presents a plan before changing anything. Use with
  /housekeeping, "housekeeping", "reduce bloat", "consolidate rules", or after
  a heavy sprint when the repo feels overgrown.
disable-model-invocation: true
---

# Housekeeping

Post-sprint cleanup — control chaos before it spreads. **Audit first, execute only after explicit user approval.**

## Hard gate

1. **Phase 1 — Report only.** No deletes, moves, edits, commits, or pushes.
2. Present findings using the report template below.
3. End with: *"Approve all / approve items N / skip / revise scope?"*
4. **Phase 2 — Execute** only the approved items, smallest diff first.

If the user said "just audit" or "report only", stop after Phase 1.

## What to scan

### Context bloat (agent surface)

- Duplicate policy across `.claude/rules/`, `AGENTS.md`, skills, README
- Multiple `alwaysApply: true` rules saying the same thing
- Skills that should be archived (one-time setup done) → `.claude/_archive/skills/`
- Stale pointers (broken skill paths, removed files still linked)
- Contradictions (e.g. one doc says run cursor rule install, another forbids it)

**Canonical pointers for this fork:**

| Topic | Single source |
|-------|----------------|
| Local code graph | `.claude/rules/local-graph.md` + `/refresh` skill |
| Agent roles | `.claude/rules/consultant-charter.md` |
| Lesson quizzes (quality) | `.claude/skills/lesson-planning/SKILL.md` |
| Batch quiz queue | `quiz-factory/` (maintainers) |
| Learner index | `.claude/skills/student-handbook/SKILL.md` |

### Folder bloat

- Untracked/generated dirs that should be gitignored (not committed): local graph cache dir, `catalog.json`, `output.txt`, etc.
- Duplicate trees (`.agents/` mirroring `.claude/` if both exist)
- Empty or single-file dirs that could fold into a parent
- Orphan artifacts from abandoned work (partial lesson dirs, duplicate capstone copies)

**Do not touch:** `phases/**/code/`, lesson content, upstream-bound PR branches, unless user explicitly includes curriculum code.

### Code & config bloat

- Dead scripts never referenced
- Overlapping automation (`scripts/` doing the same job)
- `skills-lock.json` entries for removed skills
- README/ROADMAP count drift (run `python3 scripts/check_readme_counts.py` — report only)
- Contradictory CI vs local docs

## Consolidation rules

- **One canonical home** per policy; elsewhere → one-line pointer, not a copy.
- **Merge** don't multiply: prefer extending an existing rule/skill over a new file.
- **Archive** don't delete when history helps: move to `.claude/_archive/` with `disable-model-invocation: true` and stripped trigger phrases.
- **Annihilate contradictions**: pick the fork's chosen behavior, update all references, note upstream-only exceptions separately.
- **Minimize scope**: no drive-by refactors in lesson `code/` during housekeeping.

## Phase 1 report template

```markdown
# Housekeeping audit — [date]

## Summary
- Context issues: N
- Folder issues: N
- Code/config issues: N
- Contradictions: N

## Context bloat
| Item | Location | Recommendation | Risk |
|------|----------|----------------|------|

## Folder bloat
| Item | Path | Recommendation | Risk |

## Code / config
| Item | Path | Recommendation | Risk |

## Contradictions (resolve → single truth)
| A says | B says | Chosen behavior | Files to update |

## Proposed execution order
1. …
2. …

## Out of scope (unless you ask)
- …

**Approve?** (all / pick numbers / revise / cancel)
```

Risk: **low** (docs only) | **med** (moves/archival) | **high** (deletes, curriculum, git history).

## Phase 2 execution checklist

- [ ] One logical commit per theme (not one mega-commit)
- [ ] Conventional commit: `chore(housekeeping): <what>`
- [ ] Re-run touched validators (`audit_lessons.py`, `check_readme_counts.py` if catalog touched)
- [ ] Graph refresh only if `phases/**/code/` or `site/` changed (`/refresh` skill)
- [ ] Update `student-handbook` if skills/rules moved
- [ ] Report what changed vs what was deferred

## Examples

**User:** `/housekeeping` after a 24h sprint  
**Agent:** Phase 1 report listing duplicate rule mentions, untracked junk, handbook row for removed skill. Wait.

**User:** "approve 1–3 only"  
**Agent:** Execute those three, commit, summarize.

**User:** "housekeeping but rules only"  
**Agent:** Narrow scan to `.claude/` + `AGENTS.md`, still report before edits.

## Do not

- Auto-invoke during normal feature work
- Delete `_archive/` contents without user OK
- Push without user request
- Bulk-generate quizzes or edit lesson implementations under the guise of cleanup
