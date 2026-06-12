<!-- Agent: Claude Code -->
# 00-c: Agent Setup

Write agent tailoring briefs (Echo, Newton, Lyra, Hypatia) and configure the context loader.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Workflow map | `../../CLAUDE.md` | Routing table | Scope of what each agent will do |
| Format specs | `../00-a-curriculum-archaeology/output/` | All three specs | Scope for Lyra content brief |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Full file | Scope for Lyra content and Newton briefs |
| Runtime guide | `../../references/runtime-guide.md` | Agent routing section | Declaration format |
| Student archetype | `../../vault/student-archetype.md` | Full file | Lyra content brief must be grounded in student identity |
| Course identity | `../../vault/course-identity-doc.md` | Positioning, design decisions | Lyra voice and constraint source |

## Process

1. Write Echo brief: read-only scout, BlueSkyGTM repo, 00-a archaeology scope
2. Write Newton brief: Newton's role is gap-fill research — when curriculum gaps are detected (by Hypatia at Stage 09 or by any stage discovering missing source material), Newton activates to find citations using the link repo and GLM air as the research tool. Brief must include: GTM topic clusters, GLM air as source (not Perplexity), cite every claim, stopping criteria, and the gap-fill trigger conditions (what signals Newton to activate)
3. Write Lyra content brief: ground it in vault/student-archetype.md and vault/course-identity-doc.md — the brief must name who the student is and what the course promises before specifying format constraints
4. Write Lyra code brief: site architecture constraints, render pipeline (ui.js, catalog.js), auth.js constraints, no breaking changes
5. Write Hypatia brief: curriculum coherence focus, double-helix alignment, GTM accuracy against source-citations, gap detection
6. Configure `project-keywords.json`: one keyword entry per stage so context loader auto-injects relevant files
7. Write model-config.md: GLM 5.1 for all operator-kit agents
8. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| All briefs complete | Each brief has scope, tone rules, constraints, and what NOT to do |
| Context loader covers all stages | Every build pipeline stage has at least one keyword entry |
| Invocation pattern buildable | A developer can wire Stage 08 from runtime-guide.md alone |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `agent-briefs/echo-brief.md` | `output/` | Echo tailoring brief |
| `agent-briefs/newton-brief.md` | `output/` | Newton tailoring brief |
| `agent-briefs/lyra-content-brief.md` | `output/` | Lyra content tailoring brief |
| `agent-briefs/lyra-code-brief.md` | `output/` | Lyra code tailoring brief |
| `agent-briefs/hypatia-brief.md` | `output/` | Hypatia tailoring brief |
| `project-keywords.json` | `output/` | Context loader stage keyword config |
| `model-config.md` | `output/` | GLM 5.1 assignment for all agents |
