<!-- Agent: Claude Code -->
# 00-e-full: Shared Context Vault (Complete)

Complete the vault after Helix is designed. Adds Helix voice and student promise. Supersedes seed variable registry.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| All Phase 0 outputs | `../00-a-curriculum-archaeology/output/` through `../00-d-helix-design/output/` | Voice and identity artifacts | Source for Helix voice synthesis |
| Vault seed | `../../vault/` | Existing variable-registry, course-identity-doc, student-archetype | Files to update |

## Process

1. Read Helix design outputs — synthesize voice rules from faculty-persona-spec and fsrs-integration-spec
2. Write the student promise using this scaffold: read vault/student-archetype.md (who they are, what they need, end goal). Answer: "After completing this course, [STUDENT_WHO] can [specific demonstrable capability 1], [specific demonstrable capability 2], and [specific demonstrable capability 3]." The promise must be falsifiable — a hiring manager should be able to test it in 30 minutes. Generic promises ("understand AI", "use GTM tools") are not acceptable. Derive the capabilities directly from the practitioner outcomes in 00-b's gtm-topic-map.md.
3. Update variable-registry with any new variables discovered in 00-b through 00-d. Carry forward ALL variables from the seed version — do not drop any existing resolved values when superseding.
4. Write helix-voice.md to vault
5. Run /review before closing

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `helix-voice.md` | `../../vault/` | Voice rules, tone spectrum, per-persona variation |
| `variable-registry.md` | `../../vault/` | Final complete registry — supersedes seed version |
| `course-identity-doc.md` | `../../vault/` | Updated with student promise |
