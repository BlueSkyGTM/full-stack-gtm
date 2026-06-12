<!-- Agent: Lyra -->
# Stage 02: Lesson Injection

Draft full hybrid lessons — GTM strand woven into AI engineering lessons as application layer, not parallel track.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| GTM lesson outlines | `../01-gtm-skeleton/output/gtm-lesson-outlines.md` | Full file | Outlines to expand |
| Existing AI lessons | `{{REPO_URL}}` | Target phases only | AI engineering strand to weave into |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Source citations per topic | Accuracy reference |
| Lesson manifest | `../01-gtm-skeleton/output/manifest.json` | Full file | Phase-slice queue — read status before starting, mark done after each lesson |
| Quiz factory architecture | `../../shared/quiz-factory-docs/ARCHITECTURE.md` | Manifest loop + batch pattern | Reference implementation for resumable batch processing |
| GTM handbook | `../../shared/gtm-handbook-extract.md` | Full file | Authoritative GTM content — toolstacks, KPIs, workflows, copy templates Lyra draws from when drafting GTM strand beats |
| GTM Starter Kit | `../../shared/gtm-starter-kit-guide.md` | Full file | Starter Kit skills and context file specs; exercises for Phases 01, 02, 03, 05, 17 reference the matching skill |
| MLOps appendage | `../../shared/mlops-appendage-concepts.md` | Full file | MLOps lesson content for the four post-Phase-17 appendage slots |
| Lyra content brief | `../00-c-agent-setup/output/agent-briefs/lyra-content-brief.md` | Full file | Standing orders |

## Process

1. Read manifest.json — identify all slots with status: pending, process phase by phase
2. For each phase slice:
   a. Derive the GTM application from the AI concept first — mechanically trace what the technical concept enables in a GTM context before writing any prose. If the derivation doesn't hold, flag the phase before drafting.
   b. Draft full GTM-strand content for all six beats per outline, grounded in the derivation from step (a)
   c. Weave GTM context into the corresponding AI engineering lesson — GTM as application layer
   c. Write hybrid lesson to `output/hybrid-lessons/{{PHASE}}/{{LESSON}}/docs/en.md` — include a `## Sources` block at the end with citations for every GTM claim made in the lesson (url + what it supports). Note: Tier 3 (Mermaid) diagrams written inline here will render as raw code blocks until Stage 06 confirms mermaid.js is wired on the site — this is expected and not a bug.
   d. Run all three audit checks (six-beat complete, weave-not-parallel, format match)
   e. If audit passes: mark lesson done in manifest.json and advance
   f. If audit fails: flag the lesson as blocked in manifest.json, report to human, do not advance
3. After each full phase slice completes, pause and confirm before starting the next phase

## Checkpoints

| After Step | Agent Presents | Human Decides |
|------------|---------------|---------------|
| First 5 lessons | Sample lessons from different phases | Weaving quality — approve or redirect before full run |
| End of each phase | Phase completion summary + any blocked lessons | Unblock or accept before next phase begins |

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
