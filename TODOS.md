# TODOS

Deferred scope from autoplan and stage reviews. Items here are not blocking current work.

## Loop Engineering

- [x] **Loop-engineering tooling decision — RESOLVED** — Skip npm packages. Write `.claude/skills/loop-eng-check/SKILL.md` directly (not /skillify — that produces Playwright scripts, not knowledge-injection skills). Skill conditionally loads `references/loop-engineering/` docs when loop/orchestration/sub-agent design is in scope. Wire into autoplan via CLAUDE.md routing rule.
- [ ] **Write loop-eng-check SKILL.md** — Project-local skill at `.claude/skills/loop-eng-check/SKILL.md`. Conditional check: if loop engineering isn't in the discussion (no loop/sub-agent/handler/orchestration keywords), skill skips silently. Add handlers terminology block. Include 5-in/5-out scope-detection test matrix. Wire into CLAUDE.md routing: "Loop/orchestration/sub-agent design → invoke /loop-eng-check". Defer until after Phase 0 (00-f prep).
- [ ] **Excalidraw + smart-illustrator** — Keep as Claude skills. Wire into task routing in CLAUDE.md and Lyra's content brief. No extraction or separate skill file needed.

## Operator Kit First Deployment

- [ ] **Quiz factory — 154 create_quiz rows** — Reserved as the operator kit's first real test run. Lyra (content agent) processes `quiz-factory/manifest.json` rows where `status: pending` and `job_type: create_quiz`. Covers phases 06, 07, 08, 09, 10, 12, 13, 15, 16. Manifest is clean — 128 prior rows already flipped to `done`. Audit script: `scripts/audit_lessons.py`. Batch operator rules: `shared/quiz-factory-docs/CLAUDE.md`. Run after operator kit is wired in Stage 08, as the Stage 10 validation warm-up.

## Mission Command (gtm-mission-command repo)

- [ ] **Write `references/signal-engine/`** — Reference doc (parallel to `references/loop-engineering/`) teaching the Career Ops signal pattern abstractly: signal sources, sub-agent processing, state management, batch limits. Write before Stage 06 runs. Career Ops (santifer/career-ops) is the architectural reference; this doc translates it to GTM domains (Crunchbase, BuiltWith, LinkedIn, G2).
- [ ] **Scaffold `gtm-mission-command` repo** — New repo with: `context/` (company.md, ICP.md, signals.md, playbooks/), `signals/` (scrapers/, processors/, examples/), `handlers/` (research-, score-, outreach-handler), `state/STATE.md`, `CLAUDE.md`. Include 3-5 sample signal payloads in `signals/examples/`. BATCH_SIZE + RATE_LIMIT_DELAY as env vars from day one. STATE.md prune step from day one. Create before Stage 08 runs.
- [ ] **Redesign Stage 08 CONTEXT.md** — Current scope: "verify agent wiring + extend keyword routing." New scope: wire handlers into the signal engine (research-handler, score-handler, outreach-handler deploy sub-agents against signal payloads). Maker/checker split required. Required before Stage 08 runs.
- [ ] **Add helix-ops-spec.md to Stage 05 outputs** — Operations mode for Helix post-graduation: RESEARCH/SCORE/BRIEF/DRAFT/STATUS/ESCALATE modalities. Same governed maze, different decision nodes. Reads mission-command context files (company.md, ICP.md, STATE.md). Add to Stage 05 CONTEXT.md as a required output alongside fsrs-integration-spec.md. Required before Stage 05 runs.
- [ ] **Personal fork: fenton-gtm-command** — After gtm-mission-command scaffold exists, fork as private instance for Fenton Tax. This IS the n8n/Postgres pipeline rebuilt with Claude as orchestrator. Stage 10 validation confirms the mission command can replicate what n8n currently does.

## Editor Mode (Pre-Stage-05 Requirement)

- [ ] **Write `/edit-mode` skill** — `.claude/skills/editor-mode/SKILL.md`. Must exist before Stage 05 builds Helix, so the course author can test gate logic without completing prerequisites.
  - `/edit-mode on` → creates `.editor-mode` file in the mission command worktree (gitignored — structurally cannot corrupt `progress.json`)
  - `/edit-mode off` → removes the file
  - `/edit-mode status` → reports what's bypassed
  - `/edit-mode unlock <stage>` → simulates one gate as cleared without completing prerequisites
  - Helix reads `.editor-mode` BEFORE any gate check — if present, all checks return `cleared: true`
  - All Helix responses while active MUST be prefixed with `[EDITOR MODE — gate checks bypassed]`
  - Add `.editor-mode` to `.gitignore` of the mission command scaffold (`gtm-mission-command`)
  - Add editor mode bypass to `gate-check-spec.md` (Stage 07 output)
  - **This is NOT operator mode.** Editor mode is a testing harness. Operator mode is earned by completing the course. The distinction must be explicit in the SKILL.md.

## Phase 0 / Tooling

- [ ] **Write 00-f tooling stage CONTEXT.md** — Stage for gbrain, graphify, context loader, Helix open brain setup. Deferred until operator-kit agents exist (Stage 01+). Now also needs loop engineering standards setup (LOOP.md is written; 00-f should wire it into the build pipeline operationally).
