<!-- Agent: Claude Code -->
# 00-b: GTM Content Mapping

Map GTM concepts to the existing 20-phase curriculum using a GLM air research loop.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Lesson format spec | `../00-a-curriculum-archaeology/output/lesson-format-spec.md` | Full file | Know how many beats per lesson to map into |
| 80/20 GTM Handbook | `../../shared/gtm-handbook-extract.md` | Full file | Primary GTM topic source — 10 cluster definitions |
| GTM curriculum integration | `../../shared/gtm-curriculum-integration.md` | Full file | 20-phase redirect map — starting structure for phase mapping |
| GTM integration citations | `../../shared/gtm-integration-citations.md` | Full file | 100+ phase-mapped URLs — source material for source-citations.md |
| Runtime guide | `../../references/runtime-guide.md` | Research loop section | Stopping criteria for research depth |

## Process

1. Read lesson-format-spec to understand the lesson structure that will implement each GTM cluster
2. Identify GTM topic clusters from the 80/20 Handbook: TAM Mapping, TAM Refinement, Copywriting, AI Personalization, Deliverability, Cold Email, Cold Calls, Micro Lists, News-Led Outbound, Scraping
3. For each cluster: use GLM air (Claude Code runs the research directly — Newton is not active in Phase 0; GLM air replaces Perplexity for speed and tooling affinity) to extract concepts and cite sources. Stop when each cluster has 3+ concrete techniques and 2+ examples with citations
4. Build the outcomes thesis: derive 5 practitioner outcomes (Signal machine, Write at scale, Score and qualify, Living GTM system, Agent stack) from the clusters. Each outcome is a capability a GTM engineer can demonstrate after the course — not an AI concept. Write this before the phase table.
5. Map each GTM cluster to its primary outcome and home phase — the phase table implements the outcomes thesis, not the other way around
6. Flag phases where no good GTM fit exists (exercise-only phases)
7. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| All clusters mapped | All 10 clusters have at least one phase mapping |
| No phase overloaded | No single phase carries more than 3 GTM concepts without a split note |
| Citations present | Every concept has at least one source in source-citations.md |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `gtm-topic-map.md` | `output/` | Outcomes thesis (5 practitioner outcomes) + phase table showing which phases implement which outcomes |
| `source-citations.md` | `output/` | All sourced URLs with concept attribution |
