<!-- Agent: Lyra-code -->
# Stage 05: Helix Build

Build Helix from scratch against vault/helix-architecture.md. Implement FSRS and copy-paste flag parser.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Helix design | `../00-d-helix-design/output/` | All four artifacts | Spec to implement |
| Exercise specs | `../03-exercise-design/output/exercise-specs/` | Flag format reference | Verify flag parsing |
| Quiz bank | `../04-quiz-recall/output/quiz-bank/` | Card format reference | Verify card input format |
| Lyra code brief | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | Standing orders |
| Helix architecture | `../../vault/helix-architecture.md` | Full file | Governed-maze design — build from this spec |
| Quality standards | `../../shared/quality-standards.md` | AI Interaction section | Pedagogical dimensions Helix must satisfy |
| Helix test harness | `../../vault/helix-test-harness.md` | Full file | 3-scenario test loop — must pass before output ships |

## Process

1. Build Helix system prompt from scratch using `vault/helix-architecture.md` — four layers: identity, reasoning chain, modality rules, constraints
2. Implement FSRS scheduling per fsrs-integration-spec (background layer, separate from conversation)
3. Implement copy-paste flag parser per copy-paste-flag-format (locked format — do not deviate)
4. Implement faculty persona system per faculty-persona-spec (voice register shift, not personality change)
5. Wire to quiz bank input format
6. Run the helix-test-harness.md loop — all 3 scenarios, must pass ≥ 4/5 dimensions each
7. Run /review gate before output is written

## Audit

| Check | Pass Condition |
|-------|---------------|
| Identity clean | No "Professor Synapse" references remain in system prompt or user-facing output |
| Flag parser exact | Parses the locked format string, fails gracefully on malformed input |
| FSRS wired | Card scheduling uses fsrs-integration-spec parameters |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `helix-agent/` | `output/` | Working Helix implementation |
| `helix-invocation-guide.md` | `output/` | How students invoke Helix from the terminal: command syntax, flag options, when to use Helix vs reading the site. Included in Phase 01 lesson as the course's first Helix touchpoint. |
