<!-- Agent: Hypatia -->
# Stage 10: Validation Run

Walk one complete student path end-to-end and report failures with reproduction steps.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Full curriculum | `../02-lesson-injection/output/hybrid-lessons/` | One complete student path | Post-Stage-09 lessons |
| Helix | `../05-helix-build/output/helix-agent/` | Running instance | Agent to test |
| Student state | `../07-student-state/output/` | Running implementation | Persistence to verify |
| Live site | `{{SITE_URL}}` | Post-Stage-06 deploy | Site to test against |

## Process

1. Walk one complete student path: placement → first lesson → CLI exercise → Helix recall → next lesson
2. Validate: lesson renders correctly, flag triggers Helix, FSRS schedules next review, state persists
3. Report failures with exact reproduction steps
4. Run /canary post-deploy

## Audit

| Check | Pass Condition |
|-------|---------------|
| Full path completes | Student can go from placement to second lesson without error |
| Flag triggers Helix | Copy-paste flag causes Helix to respond with correct recall card |
| State persists | Student progress survives a session reload |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `validation-report.md` | `output/` | Pass/fail per checkpoint, reproduction steps for any failure |
