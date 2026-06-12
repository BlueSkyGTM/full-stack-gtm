<!-- Agent: Claude Code -->
# 00-e-seed: Shared Context Vault (Seed)

Bootstrap the variable registry, course identity, and student archetype before agents start generating content. Student promise deferred to 00-e-full.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Variable source | `../../setup/questionnaire.md` | Full file | Variable values to resolve |
| GTM topic map | `../00-b-gtm-content-mapping/output/gtm-topic-map.md` | Full file | Student archetype derivation |

## Process

1. Resolve all `{{VARIABLE}}` placeholders from setup/questionnaire.md into a registry — include the following runtime URLs:
   - `REPO_URL` — GitHub URL or local path to the AI engineering lesson source (phases/)
   - `SITE_URL` — Live site URL for Stage 10 validation (e.g. https://learn.blueskygtm.engineer)
2. Draft the course identity doc: course name (Full-Stack GTM), positioning statement. Do not write the student promise — deferred to 00-e-full
3. Draft the student archetype: who takes this course, prior knowledge, end goal
4. Write all outputs to vault/

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `variable-registry.md` | `../../vault/` | All resolved variables |
| `course-identity-doc.md` | `../../vault/` | Course name, positioning — no student promise yet |
| `student-archetype.md` | `../../vault/` | Who the student is, prior knowledge, end goal |
