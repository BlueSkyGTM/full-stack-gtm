<!-- Agent: Newton -->
# 00-b: GTM Content Mapping

Map GTM concepts to the existing 20-phase curriculum using a Perplexity-powered research loop.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Lesson format spec | `../00-a-curriculum-archaeology/output/lesson-format-spec.md` | Full file | Know how many beats per lesson to map into |
| 80/20 GTM Handbook | `../../shared/gtm-handbook-extract.md` | Full file | Primary GTM topic source |
| Runtime guide | `../../references/runtime-guide.md` | Research loop section | Newton invocation and stopping criteria |

## Process

1. Read lesson-format-spec to understand the lesson structure Newton must map into
2. Identify GTM topic clusters from the 80/20 Handbook: TAM Mapping, TAM Refinement, Copywriting, AI Personalization, Deliverability, Cold Email, Cold Calls, Micro Lists, News-Led Outbound, Scraping
3. For each cluster: run Perplexity research loop, extract concepts, cite sources. Stop when each cluster has 3+ concrete techniques and 2+ examples with citations
4. Map each GTM concept to the existing 20-phase curriculum — which phase is the natural home
5. Flag phases where no good GTM fit exists (exercise-only phases)
6. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| All clusters mapped | All 10 clusters have at least one phase mapping |
| No phase overloaded | No single phase carries more than 3 GTM concepts without a split note |
| Citations present | Every concept has at least one source in source-citations.md |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `gtm-topic-map.md` | `output/` | GTM concept to phase mapping table with exercise potential |
| `source-citations.md` | `output/` | All sourced URLs with concept attribution |
