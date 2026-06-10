<!-- Agent: Hypatia -->
# Stage 09: Quality Pass

Audit double-helix alignment and GTM accuracy. Lyra fills identified gaps directly into Stage 02 output.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| All hybrid lessons | `../02-lesson-injection/output/hybrid-lessons/` | Full curriculum | What to audit |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Full file | Alignment reference |
| Source citations | `../00-b-gtm-content-mapping/output/source-citations.md` | Full file | GTM accuracy check |
| Helix voice | `../../vault/helix-voice.md` | Full file | Voice consistency check |

## Process

1. Audit double-helix alignment: GTM strand woven into AI lessons, not parallel
2. Audit GTM accuracy: concepts match source citations
3. Identify gaps: phases with thin or missing GTM content
4. Lyra fills gaps per gtm-topic-map guidance — writes filled lessons back to `../02-lesson-injection/output/hybrid-lessons/`

## Audit

| Check | Pass Condition |
|-------|---------------|
| Alignment score | Every phase has at least one woven GTM-AI connection |
| Accuracy verified | No GTM claim lacks a citation from source-citations.md |
| Gaps filled | All identified gaps have corresponding filled lesson files in Stage 02 output |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `curriculum-audit-report.md` | `output/` | Alignment scores, gap list, fill actions taken |
| Gap-filled lessons | `../02-lesson-injection/output/hybrid-lessons/` | Written in place — Stage 10 reads this path |
