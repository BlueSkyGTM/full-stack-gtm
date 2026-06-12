<!-- Agent: Lyra-code -->
# Stage 07: Student State

Choose and implement the student state persistence mechanism.

## The Core Model: The Repo Is the Save File

Student state does not live on the site's servers. It lives in `progress/progress.json` in the student's mission command fork. Helix writes to it. The student commits it. Their fork is the cartridge — progress travels with the repo, not with the site.

**The business registration is the new game.** Before a student can begin, they must configure `context/company.md` with a real business — their own, or one they are building. This is the save file creation moment. A repo with placeholder values in company.md is an unconfigured save file. Helix will not route operational requests against placeholder context. The student must commit to a real business to unlock the course.

This also solves the anti-cheat problem structurally: cloning someone else's completed repo gives you their business, their ICP, their scrapers targeting their signals. There is nothing to gain. Progress is only meaningful in the context of the business the student configured.

## The Naming Mechanic

The command center is called the Albatross — BlueSkyGTM's mascot. Every student starts with the same name. When operator mode is earned (Stage 10 validation complete, all gates cleared), two things unlock simultaneously and never before:

1. The student can rename their command center. Whatever they call it, that name propagates through their CLAUDE.md, STATE.md header, and Helix's self-references.
2. The student can rename Helix. Same propagation. Their instance, their co-pilot, their name.

The gate is hard. The course author does not get operator mode until they complete the validation run on their own instance. Nobody earns it by building the course — they earn it by running it. The distinction matters.

## How Locking Works

Helix reads `progress/progress.json` before routing any request. Gate checks are artifact-based, not certificate-based:

```
context/company.md — no {{PLACEHOLDERS}}?     → Stage 01 cleared
signals/scrapers/ — at least one scraper?      → Stage 06 cleared
handlers/research-handler/ — exists + valid?   → Stage 08 cleared
progress/progress.json — quiz scores logged?   → lesson-level gates
```

Helix does not say "Stage 06 is locked." It says "your research handler is trying to read from signals/processors/ but that directory is empty — looks like your signal processor isn't built yet." Then it redirects to the first missing piece. The student always knows exactly what to do next. No arbitrary walls — just the dependency graph surfaced clearly.

You cannot fake completion. An empty directory or a file with placeholder values fails the check. The component must exist and be correctly configured for the student's specific business context. The dependency graph is the anti-cheat.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Auth audit | `../00-a-curriculum-archaeology/output/auth-audit.md` | Full file | Known constraints and failure modes |
| Student state options | `../00-d-helix-design/output/student-state-options.md` | Full file | Mechanism candidates and criteria |
| Helix data model | `../05-helix-build/output/helix-agent/` | Data model | What Helix must persist |
| Lyra code brief | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | Site architecture constraints |
| Pacing map | `../06-site-readability/output/pacing-map.md` | Full file | Which gates correspond to pacing milestones |

## Process

1. Run /spec — interrogate what student state actually needs to store
2. Evaluate mechanism candidates against criteria in student-state-options
3. Implement chosen mechanism — repo-as-save-file is the primary model; site auth is secondary
4. Implement gate checks: artifact-existence + placeholder detection + quiz score thresholds
5. Implement Helix redirect logic: missing component → identify first gap → surface to student
6. Integrate with auth.js per constraints in auth-audit

## Checkpoints

| After Step | Agent Presents | Human Decides |
|------------|---------------|---------------|
| Step 2 | Mechanism evaluation matrix with recommendation | Which mechanism to implement |
| Step 4 | Gate check logic for all stages — confirm thresholds | Approve or adjust per-stage gates |

## Audit

| Check | Pass Condition |
|-------|---------------|
| Auth.js compatible | Integration does not break existing session handling |
| Helix data persists | All fields from Helix data model are stored and retrievable |
| Failure mode handled | The known failure mode from auth-audit is explicitly addressed |
| Business gate enforced | company.md with placeholders blocks operational Helix routing |
| Redirect is specific | Missing component surfaces the first gap, not a generic locked message |
| Anti-cheat is structural | Copied repo with someone else's context provides no usable state |
| Rename mechanism specified | gate-check-spec.md includes `/rename <name>` Helix command, available only when progress.json shows all Stage 10 gates cleared; propagates through CLAUDE.md, STATE.md, and Helix self-references |
| Editor mode bypass documented | gate-check-spec.md includes bypass logic: if `.editor-mode` file exists, all gate checks return cleared |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `student-state-solution/` | `output/` | Implementation and integration notes |
| `gate-check-spec.md` | `output/` | Per-stage gate conditions and redirect logic |
