# BlueSkyGTM Engineering

Build the BlueSkyGTM Engineering curriculum — GTM engineering with a technical foundation, built using the same agentic methodology it teaches.

## Operating Model — Conductor + The Loom

Two GLM squads, one ecosystem. **The Loom** builds: GLM-5.2 oversees and judges, GLM-5.1/5.1v write, the manifest is the loop. **The Tailors** finish: Echo · Newton · Turing · Shannon · Hinton (GLM-5.2) verify the AI seam and cross-reference sources; Hypatia tailors the GTM seam. The Loom weaves the cloth; the Tailors check the fit. They run on **separate manifests** and **chain sequentially** (Loom → ship-ready → Tailors → Loom correction) so they never compete for the 5-call ceiling — Tailors verify-and-flag, they never rewrite. **Route all bulk and generation work to The Loom** — lessons, exercises, quizzes, completions, regenerations, content fixes — and accuracy/cross-reference work to The Tailors. Launch via `python skills/operator-kit/orchestrator.py` (or `run.ps1 <stage>`); monitor the manifest. Architecture: `skills/operator-kit/HIERARCHY.md` · squad registry: `skills/operator-kit/TAILORS.md`.

**Your job (Conductor / Claude) is two things only:**
1. **Orchestration** — plan the work, launch The Loom, monitor the manifest, gate quality, chain stages, commit. You do **not** write curriculum content yourself; The Loom does.
2. **Personally assisting the Director** — answer questions, read and judge output, surface real decisions, design architecture, review and reason. The high-leverage, human-facing work.

~85% of token spend is The Loom (plan-covered GLM, $0). The Conductor is ~15%: orchestration + assistance. If you catch yourself generating lesson/exercise/quiz content in your own context, or spawning Claude sub-agents as Taskmasters (the deprecated hybrid pattern), stop — that work routes to The Loom. ICM/ICL is *how* the work is organized; The Loom is *who* does it; the Conductor conducts.

## Folder Map

```
full-stack-gtm/
├── CLAUDE.md                  (you are here)
├── CONTEXT.md                 (start here for task routing)
├── setup/                     (onboarding questionnaire)
├── skills/                    (gstack suite + operator-kit agents — always tracked, never gitignored)
├── vault/                     (shared context: identity, voice, variables)
├── references/
│   └── runtime-guide.md       (agent routing, gstack triggers, context loader, gbrain)
├── shared/                    (cross-stage reference files)
└── stages/
    ├── 00-a-curriculum-archaeology/
    ├── 00-b-gtm-content-mapping/
    ├── 00-c-agent-setup/
    ├── 00-d-helix-design/
    ├── 00-e-seed/
    ├── 00-e-full/
    ├── 01-gtm-skeleton/
    ├── 02-lesson-injection/
    ├── 03-exercise-design/
    ├── 04-quiz-recall/
    ├── 05-helix-build/
    ├── 06-site-readability/
    ├── 07-student-state/
    ├── 08-agent-wiring/
    ├── 09-quality-pass/
    └── 10-validation-run/
```

## Phase 0 Entry Point

**Start here for Phase 0:** `vault/phase0-plan.md` — confirms variable state, lists the 6-stage sequence, and defines human gates. Run stages in order: 00-a → 00-b → 00-e-seed → 00-c → 00-d → 00-e-full. Confirm `vault/variable-registry.md` has no `{{...}}` placeholders before starting.

## Triggers

| Keyword | Action |
|---------|--------|
| `setup` | Run onboarding questionnaire |
| `status` | Show pipeline completion for all stages |

## Routing

| Task | Go To |
|------|-------|
| Curriculum archaeology | `stages/00-a-curriculum-archaeology/CONTEXT.md` |
| GTM content mapping | `stages/00-b-gtm-content-mapping/CONTEXT.md` |
| Agent setup | `stages/00-c-agent-setup/CONTEXT.md` |
| Helix design | `stages/00-d-helix-design/CONTEXT.md` |
| Shared context (seed) | `stages/00-e-seed/CONTEXT.md` |
| Shared context (complete) | `stages/00-e-full/CONTEXT.md` |
| GTM curriculum skeleton | `stages/01-gtm-skeleton/CONTEXT.md` |
| Lesson injection | `stages/02-lesson-injection/CONTEXT.md` |
| Exercise design | `stages/03-exercise-design/CONTEXT.md` |
| Quiz and recall design | `stages/04-quiz-recall/CONTEXT.md` |
| Helix build | `stages/05-helix-build/CONTEXT.md` |
| Site readability overhaul | `stages/06-site-readability/CONTEXT.md` |
| Student state | `stages/07-student-state/CONTEXT.md` |
| Agent wiring | `stages/08-agent-wiring/CONTEXT.md` |
| Quality pass | `stages/09-quality-pass/CONTEXT.md` |
| Validation run | `stages/10-validation-run/CONTEXT.md` |

## What to Load

| Task | Load These | Do NOT Load |
|------|-----------|-------------|
| Any Phase 0 stage | `references/runtime-guide.md`, that stage's CONTEXT.md | Other stage outputs |
| Any build pipeline stage | `references/runtime-guide.md`, `vault/`, that stage's CONTEXT.md | Other stage CONTEXT.md files |
| Helix build (Stage 05) | Add `stages/00-d-helix-design/output/` | Stage 01-04 full output folders |
| Site work (Stage 06) | Add `stages/00-a-curriculum-archaeology/output/design-system-snapshot.md` | Lesson content |
| Quality pass (Stage 09) | Add `stages/00-b-gtm-content-mapping/output/source-citations.md` | Code stages output |

## Stage Handoffs

Each stage writes to its own `output/` folder. The next stage reads from there. Edit an output file and the next stage picks up your edits.

## Skill Routing

gstack skills are slash commands invoked inside Claude Code — there is no terminal `gstack` command. Full skill list and trigger table: `references/runtime-guide.md`.

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Author a backlog-ready spec / surface gaps → invoke `/spec`
- Generate missing technical or educational documentation → invoke `/document-generate`
- Full plan review / pre-stage gate → invoke `/autoplan`
- Code review / diff check → invoke `/review`
- Ship / PR → invoke `/ship`
- Bugs / errors → invoke `/investigate`
- Visual site changes → invoke `/design-review`
- Direction pivot or scope question → invoke `/office-hours`
- Post-deploy monitoring → invoke `/canary`
- Save progress → invoke `/context-save`
- Resume context → invoke `/context-restore`
- Security audit → invoke `/cso`
- Codebase health check → invoke `/health`
- ICM design / agent onboarding / new pipeline stage / multi-agent architecture → invoke `/interpreted-context-manifest`
- ICL design / recursive goal / loop that survives session boundaries / chaining stages → invoke `/interpreted-context-loop`
- Loop / orchestration / sub-agent / handler design → invoke `/loop-eng-check` *(skill not yet written — see TODOS.md; do NOT invoke until `.claude/skills/loop-eng-check/SKILL.md` exists)*
- Testing Helix gate logic as course author → invoke `/edit-mode` *(skill not yet written — see TODOS.md)*

**To rebuild gstack after pulling:** `cd ~/.claude/skills/gstack && ./setup`

### `/document-generate` — when to use in this project

This skill fills two gaps the build pipeline doesn't cover automatically:

1. **Missing exercise documentation** — when a stage produces exercises that lack a worked example, expected output, or failure-mode description, run `/document-generate` targeting that stage's `output/` folder. It will produce a how-to + reference doc pair per the Diataxis framework.
2. **Missing technical reference for live site features** — auth flow, render pipeline, catalog schema, Helix integration — any undocumented surface area in `site-new/` or `phases/` that a build agent (Lyra-code, Stage 05-08) needs but can't infer from CONTEXT.md alone.

Run it before Stage 01 if 00-a archaeology surfaces undocumented patterns. Run it after Stage 05 if Helix behavior needs a worked reference before Stage 08 wires it.

## GBrain Configuration (configured by /setup-gbrain)
- Mode: local-stdio
- Engine: pglite (`~/.gbrain/brain.pglite`)
- Config file: `~/.gbrain/config.json` (mode 0600)
- Setup date: 2026-06-12
- MCP registered: yes (user scope, via `gbrain serve`)
- Chat model: `zhipu:GLM-5.1` (Z.ai — requires `ZHIPUAI_API_KEY` in env)
- Expansion model: `zhipu:GLM-5-Turbo`
- Embedding model: `ollama:nomic-embed-text` (768d) — local, free, no API key required
  - Ollama installed 2026-06-12 via winget. Service starts automatically on login.
  - Z.ai embedding-3 was not activated in BigModel console — blocked on Z.ai side
- Artifacts sync: off (run `/sync-gbrain` to configure after Phase 0 stages produce content)
- RESOLVER.md: `skills/RESOLVER.md` (skill routing for gbrain resolver)
- Health: 65/100 (warnings only: takes_count=0, pack upgrade available)
- Restart Claude Code to pick up `mcp__gbrain__*` tools

**Note on Helix open-brain:** Not an infrastructure item. It is a STUDENT-FACING product feature (gbrain + FSRS student memory layer) built in Stage 05. Do not conflate with gstack's gbrain above.

## GBrain Search Guidance (configured by /setup-gbrain)
<!-- gstack-gbrain-search-guidance:start -->

GBrain is set up on this machine. The brain is empty until Phase 0 stages run and `/sync-gbrain` imports content. Once synced, prefer gbrain over Grep for semantic questions.

Prefer gbrain when:
- "Where is X handled?" / semantic intent: `gbrain search "<terms>"` or `gbrain query "<question>"`
- "What did we decide?" / past plans, decisions: `gbrain search "<terms>"`

Run `/sync-gbrain` after each Phase 0 stage completes to import that stage's output into the brain.

<!-- gstack-gbrain-search-guidance:end -->

---

## → Next: Operating The Loom

One shallow spine, orientation to action. Each file ends by pointing at the next; the last one runs.

```
CLAUDE.md (you are here)  →  skills/operator-kit/HIERARCHY.md  →  skills/operator-kit/orchestrator.py
   what & who                  how the stack works + quick start        the file that does the work
```

**Reference shelf** (consult, not a chain — these do not route onward):
- `skills/operator-kit/HANDLERS.md` — full tier/model/config registry
- `skills/operator-kit/TASKMASTER-PROTOCOL.md` — the GLM-5.2 orchestrator pattern
- `skills/operator-kit/interpreted-context-manifest/SKILL.md` · `interpreted-context-loop/SKILL.md` — the ICM/ICL methodology

If you are a Conductor reading this cold (e.g. after a context compaction): follow the spine. `HIERARCHY.md` lands you on the orchestrator command. Route work to The Loom; conduct, don't weave.
