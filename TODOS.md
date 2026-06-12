# TODOS

Deferred scope from autoplan and stage reviews. Items here are not blocking current work.

## Loop Engineering

- [x] **Loop-engineering tooling decision — RESOLVED** — Skip npm packages. Instead: `/scrape` the cobusgreyling substack article + `/skillify` it as a browser skill. This embeds loop engineering language + frameworks into autoplan chains with a conditional check (skip if not in scope). Handlers = Claude sub-agents that deploy other Claude sub-agents. Established language for the build pipeline and curriculum.
- [ ] **Skillify loop-engineering substack** — Run `/scrape https://cobusgreyling.substack.com/p/loop-engineering` then `/skillify`. Add conditional check: if loop engineering isn't in the discussion, skill skips. Wire into autoplan as a named reference available on demand.
- [ ] **Excalidraw + smart-illustrator** — Keep as Claude skills. Wire into task routing in CLAUDE.md and Lyra's content brief. No extraction or separate skill file needed.

## Operator Kit First Deployment

- [ ] **Quiz factory — 154 create_quiz rows** — Reserved as the operator kit's first real test run. Lyra (content agent) processes `quiz-factory/manifest.json` rows where `status: pending` and `job_type: create_quiz`. Covers phases 06, 07, 08, 09, 10, 12, 13, 15, 16. Manifest is clean — 128 prior rows already flipped to `done`. Audit script: `scripts/audit_lessons.py`. Batch operator rules: `shared/quiz-factory-docs/CLAUDE.md`. Run after operator kit is wired in Stage 08, as the Stage 10 validation warm-up.

## Phase 0 / Tooling

- [ ] **Write 00-f tooling stage CONTEXT.md** — Stage for gbrain, graphify, context loader, Helix open brain setup. Deferred until operator-kit agents exist (Stage 01+). Now also needs loop engineering standards setup (LOOP.md is written; 00-f should wire it into the build pipeline operationally).
