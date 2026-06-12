<!-- Agent: Lyra -->
# Stage 04: Quiz and Recall Design

Generate FSRS-formatted active recall card banks per phase.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Hybrid lessons | `../02-lesson-injection/output/hybrid-lessons/` | All lessons | Content to generate cards from |
| Exercise specs | `../03-exercise-design/output/exercise-specs/` | All specs | Exercise context for card tagging |
| FSRS integration spec | `../00-d-helix-design/output/fsrs-integration-spec.md` | Card schema section | Card format to write to |
| GTM handbook | `../../shared/gtm-handbook-extract.md` | KPIs and STACK sections per topic | Source for GTM-strand recall cards — tool names, KPI benchmarks, and workflow step sequences that students must retain |
| Quiz factory architecture | `../../shared/quiz-factory-docs/ARCHITECTURE.md` | Full file | Existing production schema (6 questions: pre, check×3, post×2) and pedagogical pipeline — use this, do not reinvent |
| Quiz quality rubric | `../../shared/quiz-factory-docs/QUALITY-RUBRIC.md` | Full file | Pass/Revise criteria for alignment, difficulty, technical correctness, and tone |

## Process

1. For each phase, generate an active recall question bank
2. Format each card per the FSRS card schema from fsrs-integration-spec
3. Tag each card with the lesson it reviews and the strand (AI engineering vs GTM)

Note: Stage 04 produces FSRS-formatted card files. SRS scheduling (when to surface each card) runs at Helix runtime — Stage 04 does not build or require a separate SRS backend.

## Audit

| Check | Pass Condition |
|-------|---------------|
| Schema compliance | Every card matches the FSRS card schema exactly |
| Both strands covered | Each phase has cards for both AI engineering and GTM content |
| Tags present | Every card has lesson reference and strand tag |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `quiz-bank/` | `output/` | FSRS-formatted cards, one file per phase |
