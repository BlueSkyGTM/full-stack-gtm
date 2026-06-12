<!-- Agent: Lyra-code -->
# Stage 08: Agent Wiring

Verify all stage CONTEXT.md files are correctly wired and extend project-keywords.json. Do NOT author or overwrite existing CONTEXT.md files — they are hand-curated and treated as locked inputs.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Agent briefs | `../00-c-agent-setup/output/agent-briefs/` | All five briefs | What each agent needs in its CONTEXT.md |
| Project keywords | `../00-c-agent-setup/output/project-keywords.json` | Full file | Current context loader config to extend |
| Runtime guide | `../../references/runtime-guide.md` | Agent routing section | Declaration format |
| All build pipeline artifacts | `../01-gtm-skeleton/` through `../07-student-state/output/` | Existence check | Verify outputs exist before wiring |

## Process

1. Verify all Stage 01-07 output folders contain artifacts before proceeding
2. For each build pipeline stage (01-10), READ the existing CONTEXT.md — do not overwrite
3. Verify each CONTEXT.md has: correct `<!-- Agent: -->` declaration, all input paths pointing to real output locations, no unfilled `{{VARIABLE}}` placeholders
4. Flag any missing or malformed declarations — report to human before fixing
5. Update `project-keywords.json` with a keyword entry per stage
6. Verify end-to-end for one stage: keyword fires, context loads, agent invoked, output lands at correct path

## Audit

| Check | Pass Condition |
|-------|---------------|
| All declarations present | Every build pipeline CONTEXT.md has a valid Agent declaration |
| No broken input paths | All relative paths in each CONTEXT.md resolve to existing files |
| No unresolved placeholders | No `{{VARIABLE}}` strings remain in any CONTEXT.md |
| Keywords complete | Every stage has at least one entry in project-keywords.json |
| One stage verified | End-to-end test passes for one stage before rest are wired |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Wiring audit report | `output/wiring-audit.md` | Per-stage: declaration status, broken paths, placeholder gaps |
| `project-keywords.json` | `../00-c-agent-setup/output/` (updated in place) | Stage keyword entries added |
