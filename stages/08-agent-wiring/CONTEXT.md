<!-- Agent: Lyra-code -->
# Stage 08: Agent Wiring

Write CONTEXT.md files at canonical stage paths and update project-keywords.json.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Agent briefs | `../00-c-agent-setup/output/agent-briefs/` | All five briefs | What each agent needs in its CONTEXT.md |
| Project keywords | `../00-c-agent-setup/output/project-keywords.json` | Full file | Current context loader config to extend |
| Runtime guide | `../../references/runtime-guide.md` | Agent routing section | Declaration format |
| All build pipeline artifacts | `../01-gtm-skeleton/` through `../07-student-state/output/` | Existence check | Verify outputs exist before wiring |

## Process

1. Verify all Stage 01-07 output folders contain artifacts before proceeding
2. For each build pipeline stage (01-10), write the stage CONTEXT.md at its canonical path with the correct `<!-- Agent: -->` declaration
3. Wire each CONTEXT.md to reference the correct prior stage output paths
4. Update `project-keywords.json` with a keyword entry per stage
5. Verify end-to-end for one stage: keyword fires, context loads, agent invoked, output lands at correct path
6. CONTEXT.md files are written to their actual stage folders, not staged in output/

## Audit

| Check | Pass Condition |
|-------|---------------|
| All declarations present | Every build pipeline CONTEXT.md has a valid Agent declaration |
| Keywords complete | Every stage has at least one entry in project-keywords.json |
| One stage verified | End-to-end test passes for one stage before rest are wired |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Stage CONTEXT.md files | Canonical stage paths | Agent declaration + input paths + process brief |
| `project-keywords.json` | `../00-c-agent-setup/output/` (updated in place) | Stage keyword entries added |
