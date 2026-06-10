<!-- Agent: Claude Code -->
# 00-c: Agent Setup

Install operator-kit agents, write tailoring briefs, configure context loader.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Workflow map | `../../CLAUDE.md` | Routing table | Scope of what each agent will do |
| Format specs | `../00-a-curriculum-archaeology/output/` | All three specs | Scope for Lyra content brief |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Full file | Scope for Newton brief |
| Operator-kit | `../../skills/operator-kit/` | Agent definitions | Base agents to tailor |
| Runtime guide | `../../references/runtime-guide.md` | Agent routing section | Declaration format |

## Process

1. Install operator-kit agents into `../../skills/operator-kit/` if not present
2. Write Echo brief: read-only scout, BlueSkyGTM repo, 00-a archaeology scope
3. Write Newton brief: GTM topic clusters, Perplexity as source, cite every claim, stopping criteria
4. Write Lyra content brief: lesson format spec path, GTM tone, Full-Stack GTM identity, token budget, chunking strategy for 498-lesson scale
5. Write Lyra code brief: site architecture constraints, render pipeline (ui.js, catalog.js), auth.js constraints, no breaking changes
6. Write Hypatia brief: curriculum coherence focus, double-helix alignment, GTM accuracy against source-citations, gap detection
7. Configure `project-keywords.json`: one keyword entry per stage so context loader auto-injects relevant files
8. Write model-config.md: GLM 5.1 for all operator-kit agents
9. Run audit checks

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
