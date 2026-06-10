<!-- Agent: Lyra -->
# Stage 02: Lesson Injection

Draft full hybrid lessons — GTM strand woven into AI engineering lessons as application layer, not parallel track.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| GTM lesson outlines | `../01-gtm-skeleton/output/gtm-lesson-outlines.md` | Full file | Outlines to expand |
| Existing AI lessons | `{{REPO_URL}}` | Target phases only | AI engineering strand to weave into |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Source citations per topic | Accuracy reference |
| Lyra content brief | `../00-c-agent-setup/output/agent-briefs/lyra-content-brief.md` | Full file | Standing orders |

## Process

1. For each GTM outline, draft full GTM-strand content for all six beats
2. Weave GTM context into the corresponding AI engineering lesson — GTM as application layer
3. Write hybrid lesson to `output/hybrid-lessons/{{PHASE}}/{{LESSON}}/docs/en.md`
4. Run audit checks per lesson before moving to next

## Checkpoints

| After Step | Agent Presents | Human Decides |
|------------|---------------|---------------|
| First 5 lessons | Sample lessons from different phases | Weaving quality — approve or redirect before full run |

## Audit

| Check | Pass Condition |
|-------|---------------|
| Six-beat complete | Every lesson has all six beats with substantive content |
| Weave not parallel | GTM section names the specific AI concept it applies to in its first sentence |
| Format match | Heading levels and front-matter match lesson-format-spec |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `hybrid-lessons/` | `output/` | Full lesson drafts, one per slot, matching existing path structure |
