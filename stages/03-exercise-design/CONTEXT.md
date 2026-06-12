<!-- Agent: Lyra -->
# Stage 03: Exercise Design

Write CLI exercise specs per lesson with the LOCKED copy-paste flag embedded.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Hybrid lessons | `../02-lesson-injection/output/hybrid-lessons/` | All lessons | Lessons to write exercises for |
| Copy-paste flag format | `../00-d-helix-design/output/copy-paste-flag-format.md` | Full file | LOCKED — do not deviate |
| Exercise format spec | `../00-a-curriculum-archaeology/output/exercise-format-spec.md` | Full file | Structure and output shape |
| GTM Starter Kit | `../../shared/gtm-starter-kit-guide.md` | Skills table + Quick Start | Phases 01, 02, 03, 05, 17: exercise instructs the student to run the matching GTM Starter Kit skill against their own domain |

## Process

1. For each hybrid lesson, write the CLI exercise spec
2. Embed the exact copy-paste flag at the exercise completion point
3. Exercise must be completable in Claude Code without leaving the terminal
4. Run audit checks

## Audit

| Check | Pass Condition |
|-------|---------------|
| Flag embedded exactly | Every exercise uses the exact string from copy-paste-flag-format.md |
| Terminal-completable | No exercise requires a browser or external tool |
| Format match | Structure matches exercise-format-spec |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `exercise-specs/` | `output/` | One exercise spec per lesson with embedded copy-paste flag |
