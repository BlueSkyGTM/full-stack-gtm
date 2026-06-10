# Full-Stack GTM

Build the Full-Stack GTM course pipeline from archaeology through validation.

## Task Routing

| Task Type | Go To | Description |
|-----------|-------|-------------|
| Archaeology | `stages/00-a-curriculum-archaeology/CONTEXT.md` | Map existing site and codebase |
| GTM research | `stages/00-b-gtm-content-mapping/CONTEXT.md` | Map GTM concepts to phases via Perplexity |
| Agent setup | `stages/00-c-agent-setup/CONTEXT.md` | Install and tailor operator-kit agents |
| Helix design | `stages/00-d-helix-design/CONTEXT.md` | FSRS spec, flag format, persona system |
| Shared context | `stages/00-e-seed/CONTEXT.md` | Bootstrap variable registry and course identity |
| Content generation | `stages/01-gtm-skeleton/CONTEXT.md` | GTM lesson outlines |
| Lesson injection | `stages/02-lesson-injection/CONTEXT.md` | Full hybrid lesson drafts |
| Exercises | `stages/03-exercise-design/CONTEXT.md` | CLI exercise specs with copy-paste flags |
| Quiz bank | `stages/04-quiz-recall/CONTEXT.md` | FSRS-formatted recall cards |
| Helix agent | `stages/05-helix-build/CONTEXT.md` | Fork Synapse, implement FSRS + flag parser |
| Site redesign | `stages/06-site-readability/CONTEXT.md` | Modular lesson components |
| Student state | `stages/07-student-state/CONTEXT.md` | Persistence mechanism |
| Agent wiring | `stages/08-agent-wiring/CONTEXT.md` | Wire CONTEXT.md files + context loader |
| Quality audit | `stages/09-quality-pass/CONTEXT.md` | Coherence audit and gap fill |
| Validation | `stages/10-validation-run/CONTEXT.md` | End-to-end student path test |

## Shared Resources

| Resource | Location | Contains |
|----------|----------|----------|
| Runtime guide | `references/runtime-guide.md` | Agent routing, gstack triggers, context loader, gbrain |
| Vault | `vault/` | Course identity, variable registry, Helix voice, student archetype |
| Shared refs | `shared/` | Cross-stage files: lesson format spec, exercise format spec |
| gstack | `skills/gstack/` | /spec, /review, /ship, /design-review, /office-hours, /canary |
| Operator-kit | `skills/operator-kit/` | Lyra, Newton, Echo, Hypatia agent definitions |
