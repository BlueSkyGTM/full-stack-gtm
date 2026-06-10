<!-- Agent: Lyra-code -->
# Stage 07: Student State

Choose and implement the student state persistence mechanism.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Auth audit | `../00-a-curriculum-archaeology/output/auth-audit.md` | Full file | Known constraints and failure modes |
| Student state options | `../00-d-helix-design/output/student-state-options.md` | Full file | Mechanism candidates and criteria |
| Helix data model | `../05-helix-build/output/helix-agent/` | Data model | What Helix must persist |
| Lyra code brief | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | Site architecture constraints |

## Process

1. Run /spec — interrogate what student state actually needs to store
2. Evaluate mechanism candidates against criteria in student-state-options
3. Implement chosen mechanism
4. Integrate with auth.js per constraints in auth-audit

## Checkpoints

| After Step | Agent Presents | Human Decides |
|------------|---------------|---------------|
| Step 2 | Mechanism evaluation matrix with recommendation | Which mechanism to implement |

## Audit

| Check | Pass Condition |
|-------|---------------|
| Auth.js compatible | Integration does not break existing session handling |
| Helix data persists | All fields from Helix data model are stored and retrievable |
| Failure mode handled | The known failure mode from auth-audit is explicitly addressed |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `student-state-solution/` | `output/` | Implementation and integration notes |
