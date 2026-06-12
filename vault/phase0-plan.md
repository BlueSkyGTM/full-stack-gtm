<!-- /autoplan restore point: /Users/raymo/.gstack/projects/BlueSkyGTM-full-stack-gtm/main-autoplan-restore-20260611-215530.md -->
# BlueSkyGTM Engineering — Phase 0 Plan: Build the Ecosystem Before Releasing the Finish

## Decision

Run Phase 0 entirely with Claude Code. GLM agents (Lyra, Newton, Hypatia, Echo) activate at Stage 01.

**Why:**
- 00-a and 00-b were labeled Echo/Newton in the contracts, but agent setup (00-c) comes after them — you can't use an agent before it's configured.
- Phase 0 outputs are one-shot cold reads and design decisions, not token-burn volume work. No GLM offload needed.
- 00-c tailoring briefs require 00-a format specs and 00-b topic map as inputs. Order must hold.

## Revised Phase 0 Sequence

```
00-a  Claude Code   Read live site + repo → format specs, auth audit, design snapshot
  ↓
00-b  Claude Code   Map GTM topics to phases → gtm-topic-map, source-citations
  ↓
00-e-seed  Claude Code   Bootstrap vault → variable-registry, course-identity-doc, student-archetype
  ↓
00-c  Claude Code   Write agent briefs → agent-briefs/, project-keywords.json, model-config.md
  ↓
00-d  Claude Code   Design Helix → LOCKED: copy-paste-flag-format, fsrs-integration-spec
  ↓
00-e-full  Claude Code   Complete vault → helix-voice, updated variable-registry, student-promise
```

GLM agents go live at Stage 01 (first place Lyra writes 498 lesson outlines).

## What Each Stage Produces

### 00-a — Curriculum Archaeology
Inputs: live site (https://learn.blueskygtm.engineer), GitHub repo (phases/, site-new/, scripts/)
Outputs (to stages/00-a-curriculum-archaeology/output/):
- lesson-format-spec.md — exact six-beat structure with concrete examples
- exercise-format-spec.md — CLI exercise pattern with copy-paste flag placeholder
- quiz-format-spec.md — card schema and tag conventions
- auth-audit.md — current auth mechanism, named failure mode
- design-system-snapshot.md — colors, typography, component patterns

### 00-b — GTM Content Mapping
Inputs: lesson-format-spec.md, shared/gtm-handbook-extract.md (10 topic clusters)
Outputs (to stages/00-b-gtm-content-mapping/output/):
- gtm-topic-map.md — each cluster mapped to phase slots, max 3 concepts per phase
- source-citations.md — ≥2 cited examples per cluster

### 00-e-seed — Vault Bootstrap
Inputs: setup/questionnaire.md (resolved variable values), gtm-topic-map.md
Outputs (written directly to vault/):
- variable-registry.md — all {{VARIABLES}} resolved
- course-identity-doc.md — Full-Stack GTM positioning, NO student promise yet
- student-archetype.md — STUDENT_WHO, PRIOR_KNOWLEDGE, NEED, END_GOAL

### 00-c — Agent Setup
Inputs: all 00-a format specs, gtm-topic-map.md, runtime-guide.md agent routing section
Outputs (to stages/00-c-agent-setup/output/):
- agent-briefs/echo-brief.md
- agent-briefs/newton-brief.md
- agent-briefs/lyra-content-brief.md
- agent-briefs/lyra-code-brief.md
- agent-briefs/hypatia-brief.md
- project-keywords.json — context loader keyword map
- model-config.md — GLM 5.1 for all agents

### 00-d — Helix Design
Inputs: exercise/lesson format specs, lyra-code-brief, fsrs-algorithm reference, vault/helix-architecture.md
Outputs (LOCKED — to stages/00-d-helix-design/output/):
- fsrs-integration-spec.md — concrete parameter values, not ranges
- copy-paste-flag-format.md — DEPRECATED 2026-06-12 (artifact-based verification replaces the flag entirely)
- faculty-persona-spec.md — Helix identity, voice constraints
- student-state-options.md — mechanism evaluation, RESOLVED: repo-as-save-file
- helix-ramp-schedule.md — Zone activation schedule, Zone 1-3 standard Claude, Zone 4+ full Helix

### 00-e-full — Vault Complete
Inputs: all Phase 0 outputs, existing vault/ files
Outputs (written directly to vault/):
- helix-voice.md — hard constraints, sentence rules, tone spectrum, per-persona variation
- variable-registry.md — updated with vars from 00-b through 00-d
- course-identity-doc.md — updated with student promise

## Contracts to Patch

Both contracts already patched — 00-a and 00-b CONTEXT.md files already declare `<!-- Agent: Claude Code -->`. No action needed.

## Out of Scope
- Running any build pipeline stage (01-10) before Phase 0 completes
- Filling agent brief content (00-c writes the briefs from 00-a/00-b outputs)
- Any lesson content editing

## Success Criteria
Phase 0 is done when:
1. All 5 output folders have files (not just .gitkeep)
2. vault/ and all stage CONTEXT.md files have no unfilled {{VARIABLE}} placeholders — includes {{REPO_URL}}, {{SITE_URL}}
3. copy-paste-flag-format.md exists and contains the exact flag string
4. project-keywords.json covers all 10 build pipeline stages
5. Stage 01 dry-run passes: run 00-c's lyra-content-brief through Stage 01 CONTEXT.md against a single Phase 01 lesson slot. If Lyra produces a structurally valid outline, Phase 0 is done. If it fails (missing context, unfilled variables, structural error), Phase 0 is not done.

## Human Gates
Phase 0 has two mandatory human sign-off points before proceeding:

**Gate 1 — after 00-d:** Review `student-state-options.md` before running 00-e-full. The Helix voice and vault completion depend on this architectural decision. Do not run 00-e-full until the student state mechanism is decided.

**Gate 2 — after 00-e-full:** Review `vault/helix-voice.md` and `vault/course-identity-doc.md` (with student promise) before running Stage 01. These are the editorial foundation for 498 lessons.
✅ CLOSED 2026-06-12 — helix-voice.md: 8 hard constraints, two registers, zone-specific behavior, tone spectrum. Student promise: 3 falsifiable capabilities demonstrable in <30 min to a technical hiring manager.

## Re-run Safety
Each stage is safe to re-run IF you delete the output folder contents first (not the .gitkeep).

LOCKED outputs (00-d only): `copy-paste-flag-format.md` and `fsrs-integration-spec.md` are permanent locks. Before overwriting either, confirm explicitly: "I am intentionally relocking — all downstream Stage 05 exercises must be rebuilt." The audit check in 00-d enforces this by name.

00-e-seed and 00-e-full both write to vault/. 00-e-full supersedes 00-e-seed's variable-registry.md. If you re-run 00-e-seed after 00-e-full, you will lose any variables 00-e-full added. Sequence must hold.

---

<!-- AUTONOMOUS DECISION LOG -->
## Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|----------------|-----------|-----------|----------|
| 1 | CEO | Keep 00-b producing gtm-topic-map + source-citations (not reduce to citations-only) | Mechanical | P3 Pragmatic | Topic map is low-effort and gives Lyra grounding at Stage 01. Not worth cutting. | Collapse to citations-only |
| 2 | CEO | 00-c runs before 00-d (sequential, not parallel) | Mechanical | P5 Explicit | 00-d explicitly lists lyra-code-brief.md as input — 00-c IS a dependency | Parallelize 00-c and 00-d |
| 3 | CEO | Keep 00-e-seed as separate stage | Mechanical | P5 Explicit | Isolates vault bootstrap cleanly; agent briefs need resolved variables from vault | Merge into 00-e-full |
| 4 | CEO/Eng | Add FSRS validation example to 00-d before locking | Mechanical | P1 Completeness | 498 exercises built on unvalidated FSRS params is unacceptable risk; worked example is minimal validation | Lock without validation |
| 5 | CEO | Add explicit human gate after 00-d | Mechanical | P1 Completeness | student-state-options.md is an architecture decision; 00-e-full cannot proceed without it being resolved | Treat 00-d→00-e-full as automatic |
| 6 | CEO | Strengthen success criterion #5 to require Stage 01 dry-run | Mechanical | P1 Completeness | File-existence check is not sufficient; Phase 0 is only done when a downstream stage can actually run | Keep SC #5 as file-existence check |
| 7 | Eng | Add vault/ files to 00-c inputs | Mechanical | P1 Completeness | Lyra content brief needs student archetype and course identity to be grounded | Leave 00-c inputs incomplete |
| 8 | Eng | Fix 00-b process step 3 to say "Claude Code runs Perplexity directly" | Mechanical | P5 Explicit | Newton is retired; ghost dependency creates confusion | Leave Newton reference |
| 9 | Eng | Fix 00-c process step numbering gap (2 missing) | Mechanical | P5 Explicit | Numbering gap signals dropped step; clean it up | Leave gap |
| 10 | DX | Add Phase 0 entry point block to CLAUDE.md | Mechanical | P5 Explicit | Entry point was invisible; 1-line fix unblocks new operators | Leave it undiscoverable |
| 11 | DX | Add failure modes table to 00-a CONTEXT.md | Mechanical | P1 Completeness | Site-down is a known failure mode with no guidance; operators need recovery path | Leave silent failure |
| 12 | DX | Add re-run safety section to plan | Mechanical | P5 Explicit | No re-run protocol causes data loss risk (00-e-seed overwriting 00-e-full) | Leave undocumented |
| 13 | DX | Add student promise scaffold to 00-e-full | Mechanical | P1 Completeness | Cold-writing the hardest editorial decision produces generic output | Write promise without scaffold |
| 14 | Deferred | 00-a spec refresh mechanism at Stage 06 | Deferred → TODOS.md | P3 Pragmatic | Stage 06 scope, outside Phase 0; low risk before Stage 06 runs | Add to Phase 0 |

## TODOS.md Deferrals

- **00-a spec refresh gate (Stage 06):** Before Stage 06 runs, re-run the 00-a audit checks against the live site to catch rendering stack drift. Stage 06 CONTEXT.md should reference the git hash recorded in design-system-snapshot.md.
- **Agent wiring smoke test (Stage 08):** Add an integration check to Stage 08 that confirms project-keywords.json correctly injects files for each keyword before wiring agents.

## GSTACK REVIEW REPORT (Session 1 — 2026-06-11)

**Phases run:** CEO, Eng, DX (no UI scope detected)
**Dual voices:** Claude subagent only (Codex not installed — single-reviewer mode)
**Auto-decided:** 14 decisions
**Taste decisions surfaced at gate:** 1
**User challenges:** 0
**Critical fixes applied:** FSRS validation gate, Stage 01 dry-run SC, 00-d human gate
**High fixes applied:** vault inputs to 00-c, 00-a failure modes, student promise scaffold, CLAUDE.md entry point, re-run safety
**Deferred to TODOS.md:** 00-a spec refresh, Stage 08 wiring smoke test

## GSTACK REVIEW REPORT (Session 2 — 2026-06-11, gamification + editor mode)

**Phases run:** CEO, Design, Eng, DX
**Dual voices:** [codex-unavailable] — single-reviewer mode
**Auto-decided:** 10 decisions
**Taste decisions surfaced at gate:** 0
**User challenges:** 0
**Critical fixes applied:**
  - Stage 06 pacing-map sequencing bug fixed (Stage 02 vs Stage 09 as consumer)
  - Editor mode architecture defined (.editor-mode file, gitignored, Helix reads first)
  - helix-ramp-schedule.md added to Stage 00-d required outputs + audit check
**High fixes applied:**
  - /edit-mode added to TODOS.md as pre-Stage-05 requirement
  - Rename mechanism (Stage 07) added to audit checks
  - Editor mode bypass added to Stage 07 gate-check-spec audit check
  - loop-eng-check + /edit-mode routing stubs added to CLAUDE.md
**Design gaps identified:**
  - Progress UI ↔ repo sync mechanism deferred to Stage 07 student-state-options.md
  - [EDITOR MODE] Helix prefix specified as required indicator
**Deferred to TODOS.md:** editor mode SKILL.md (pre-Stage-05), loop-eng-check routing confirmed
