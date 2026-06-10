# Full-Stack GTM

Build the Full-Stack GTM course — a curriculum proving the convergence of AI engineering and GTM engineering, built using the same agentic methodology it teaches.

## Folder Map

```
full-stack-gtm/
├── CLAUDE.md                  (you are here)
├── CONTEXT.md                 (start here for task routing)
├── setup/                     (onboarding questionnaire)
├── skills/                    (gstack + operator-kit agents)
├── vault/                     (shared context: identity, voice, variables)
├── references/
│   └── runtime-guide.md       (agent routing, gstack triggers, context loader, gbrain)
├── shared/                    (cross-stage reference files)
└── stages/
    ├── 00-a-curriculum-archaeology/
    ├── 00-b-gtm-content-mapping/
    ├── 00-c-agent-setup/
    ├── 00-d-helix-design/
    ├── 00-e-seed/
    ├── 00-e-full/
    ├── 01-gtm-skeleton/
    ├── 02-lesson-injection/
    ├── 03-exercise-design/
    ├── 04-quiz-recall/
    ├── 05-helix-build/
    ├── 06-site-readability/
    ├── 07-student-state/
    ├── 08-agent-wiring/
    ├── 09-quality-pass/
    └── 10-validation-run/
```

## Triggers

| Keyword | Action |
|---------|--------|
| `setup` | Run onboarding questionnaire |
| `status` | Show pipeline completion for all stages |

## Routing

| Task | Go To |
|------|-------|
| Curriculum archaeology | `stages/00-a-curriculum-archaeology/CONTEXT.md` |
| GTM content mapping | `stages/00-b-gtm-content-mapping/CONTEXT.md` |
| Agent setup | `stages/00-c-agent-setup/CONTEXT.md` |
| Helix design | `stages/00-d-helix-design/CONTEXT.md` |
| Shared context (seed) | `stages/00-e-seed/CONTEXT.md` |
| Shared context (complete) | `stages/00-e-full/CONTEXT.md` |
| GTM curriculum skeleton | `stages/01-gtm-skeleton/CONTEXT.md` |
| Lesson injection | `stages/02-lesson-injection/CONTEXT.md` |
| Exercise design | `stages/03-exercise-design/CONTEXT.md` |
| Quiz and recall design | `stages/04-quiz-recall/CONTEXT.md` |
| Helix build | `stages/05-helix-build/CONTEXT.md` |
| Site readability overhaul | `stages/06-site-readability/CONTEXT.md` |
| Student state | `stages/07-student-state/CONTEXT.md` |
| Agent wiring | `stages/08-agent-wiring/CONTEXT.md` |
| Quality pass | `stages/09-quality-pass/CONTEXT.md` |
| Validation run | `stages/10-validation-run/CONTEXT.md` |

## What to Load

| Task | Load These | Do NOT Load |
|------|-----------|-------------|
| Any Phase 0 stage | `references/runtime-guide.md`, that stage's CONTEXT.md | Other stage outputs |
| Any build pipeline stage | `references/runtime-guide.md`, `vault/`, that stage's CONTEXT.md | Other stage CONTEXT.md files |
| Helix build (Stage 05) | Add `stages/00-d-helix-design/output/` | Stage 01-04 full output folders |
| Site work (Stage 06) | Add `stages/00-a-curriculum-archaeology/output/design-system-snapshot.md` | Lesson content |
| Quality pass (Stage 09) | Add `stages/00-b-gtm-content-mapping/output/source-citations.md` | Code stages output |

## Stage Handoffs

Each stage writes to its own `output/` folder. The next stage reads from there. Edit an output file and the next stage picks up your edits.
