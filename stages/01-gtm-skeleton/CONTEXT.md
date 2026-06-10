<!-- Agent: Lyra -->
# Stage 01: GTM Curriculum Skeleton

Generate one GTM lesson outline per mapped phase slot. Outlines only — no full prose.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Lesson format spec | `../00-a-curriculum-archaeology/output/lesson-format-spec.md` | Full file | Six-beat structure to use |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Full file | Which concept maps to which phase |
| Variable registry | `../../vault/variable-registry.md` | Full file | Placeholder resolution |
| Lyra content brief | `../00-c-agent-setup/output/agent-briefs/lyra-content-brief.md` | Full file | Standing orders |
| Course identity | `../../vault/course-identity-doc.md` | Full file | Positioning context |

## Process

1. For each phase in the GTM topic map, generate one GTM lesson outline per mapped slot
2. Each outline: phase number, lesson title (GTM strand), six-beat headings, GTM topic reference, exercise hook
3. Do not draft full lesson content — beat headings and exercise hooks only
4. Flag phases where GTM content does not weave (exercise-only phases)
5. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| Format compliance | Every outline uses the six-beat heading structure from lesson-format-spec |
| Full coverage | Every GTM topic from gtm-topic-map has at least one outline |
| No full drafts | Outlines have beat headings and exercise hooks, not full prose |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `gtm-lesson-outlines.md` | `output/` | One outline block per GTM lesson slot; exercise-only flags where applicable |
