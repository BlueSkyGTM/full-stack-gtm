<!-- Agent: Echo -->
# 00-a: Curriculum Archaeology

Map the existing site, codebase, and auth layer before anything is built.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Live site | `{{SITE_URL}}` | All 20 phases, lesson structure | Canonical lesson format |
| GitHub repo | `{{REPO_URL}}` | LESSON_TEMPLATE.md, one lesson per phase, all JS | Format spec source |
| Workflow map | `../../references/runtime-guide.md` | Full file | Invocation mechanics |

## Process

1. Map all 20 phases: name, lesson count, topic range
2. Extract the six-beat lesson structure from LESSON_TEMPLATE.md
3. Document the exercise format: CLI-facing, expected output shape
4. Document the quiz format: question types, scoring model
5. Read auth.js: session handling, persistence mechanism, known failure modes
6. Capture CSS/design system: class naming, component patterns, lesson layout markup
7. Note placement test mechanism if present
8. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| Format specs complete | All three specs have concrete examples, not just descriptions |
| Auth audit actionable | Names the specific failure mode and what Stage 07 must handle |
| Design snapshot has markup | Includes actual class names and HTML structure |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `lesson-format-spec.md` | `output/` | Six-beat structure, heading levels, front-matter schema |
| `exercise-format-spec.md` | `output/` | CLI exercise structure, expected output, copy-paste trigger pattern |
| `quiz-format-spec.md` | `output/` | Question types, answer formats, scoring fields |
| `auth-audit.md` | `output/` | auth.js behavior, session model, failure modes, Stage 07 implications |
| `design-system-snapshot.md` | `output/` | CSS patterns, component list, lesson layout markup |
