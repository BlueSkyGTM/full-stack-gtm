<!-- Agent: Lyra-code -->
# Stage 05: Helix Build

Fork Professor Synapse, swap identity to Helix, implement FSRS and copy-paste flag parser.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Helix design | `../00-d-helix-design/output/` | All four artifacts | Spec to implement |
| Exercise specs | `../03-exercise-design/output/exercise-specs/` | Flag format reference | Verify flag parsing |
| Quiz bank | `../04-quiz-recall/output/quiz-bank/` | Card format reference | Verify card input format |
| Lyra code brief | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | Standing orders |
| Professor Synapse | `{{SYNAPSE_REPO_URL}}` | Full repo | Fork base |

## Process

1. Fork Professor Synapse into `output/helix-agent/`
2. Swap identity: Professor Synapse to Helix — naming, voice, persona system
3. Implement FSRS scheduling per fsrs-integration-spec
4. Implement copy-paste flag parser per copy-paste-flag-format (locked format — do not deviate)
5. Implement faculty persona generation per faculty-persona-spec
6. Wire to quiz bank input format
7. Run /review gate before output is written

## Audit

| Check | Pass Condition |
|-------|---------------|
| Identity swap complete | No "Professor Synapse" references remain in user-facing output |
| Flag parser exact | Parses the locked format string, fails gracefully on malformed input |
| FSRS wired | Card scheduling uses fsrs-integration-spec parameters |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `helix-agent/` | `output/` | Working Helix implementation |
