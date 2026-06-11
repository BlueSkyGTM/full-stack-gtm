# Full-Stack GTM Engineering

A 502-lesson AI engineering course with GTM as the application layer — and the reproducible pipeline that built it.

This repository is two things. The first is a course: structured, quality-audited, AI-tutored curriculum teaching AI engineering concepts with GTM (go-to-market) as the derived application layer in every lesson. The second is the coursemaker: a domain-agnostic build pipeline that can produce the same structure for any practitioner field (security, healthcare, legal tech, finance) by swapping the application layer.

**Start here:** [COURSEMAKER.md](COURSEMAKER.md) explains the Double Helix design principle, the pipeline, and why a 3-month programmer can build this.

---

## What's Inside

```
full-stack-gtm/
├── COURSEMAKER.md             ← How this was built and why it works
├── CLAUDE.md                  ← Agent routing and stage map
├── CONTEXT.md                 ← Task routing (start here)
├── vault/                     ← Course identity, Helix architecture, voice, variables
├── shared/                    ← Cross-stage reference: handbook, quality standards, illustrations
├── stages/                    ← Build pipeline (Stages 00–10)
│   ├── 00-a-curriculum-archaeology/
│   ├── 00-b-gtm-content-mapping/
│   ├── 00-c-agent-setup/
│   ├── 00-d-helix-design/
│   ├── 00-e-seed/
│   ├── 00-e-full/
│   ├── 01-gtm-skeleton/
│   ├── 02-lesson-injection/
│   ├── 03-exercise-design/
│   ├── 04-quiz-recall/
│   ├── 05-helix-build/
│   ├── 06-site-readability/
│   ├── 07-student-state/
│   ├── 08-agent-wiring/
│   ├── 09-quality-pass/
│   └── 10-validation-run/
├── phases/                    ← AI engineering lesson content (Phases 00–19)
├── glossary/                  ← 277-term AI engineering glossary
└── antilibrary/               ← Curated cognitive anchor library
```

---

## The Build Pipeline

| Stage | What It Does |
|-------|--------------|
| 00-a–e | Archaeology, topic mapping, agent briefing, Helix design, vault bootstrap |
| 01 | Generate 502 lesson outlines + manifest (resumable queue) |
| 02 | Inject full hybrid lessons — chain-of-reason derivation, GTM weave, inline citations |
| 03 | Design CLI exercises that run entirely in Claude Code Desktop |
| 04 | Generate FSRS-formatted recall cards |
| 05 | Build Helix (the AI tutor) from the governed-maze spec |
| 06 | Site readability overhaul + 3-tier illustration generation |
| 07 | Student state persistence |
| 08 | Verify stage wiring and extend keyword routing |
| 09 | Quality pass: CLARITY, WEAVE, ACCURACY judges on all 502 lessons |
| 10 | End-to-end validation — first-15-minutes onboarding through full student path |

---

## The Double Helix

Every lesson enforces one rule: the GTM application must be mechanically derived from the AI concept, not adjacent to it. This produces a curriculum with two access paths — navigate by AI concept or by GTM use case. See [COURSEMAKER.md](COURSEMAKER.md) for the full explanation.

---

## Helix

Helix is the AI tutor. It is a governed maze — a structured decision tree with seven modalities (EXPLAIN, QUIZ, HINT, FLAG CHECK, CORRECT, ORIENT, REDIRECT). It shows its reasoning at every step. Built from scratch against the spec in [vault/helix-architecture.md](vault/helix-architecture.md).

Platform: Claude Code Desktop (terminal-only). All exercises run in your terminal.

---

## Quality

Three mechanisms enforce quality before human review:

1. **Derivation gate** — GTM application must be derived before prose is written
2. **Sources block** — every GTM claim requires an inline citation at write time
3. **Judge layer** — CLARITY, WEAVE, and ACCURACY judges run on all 502 lessons at Stage 09

Standards: [shared/quality-standards.md](shared/quality-standards.md)

---

## Running the Pipeline

1. Open Claude Code in this directory
2. Type `status` to see pipeline completion
3. Type `setup` to run onboarding
4. Navigate to a stage: `stages/00-a-curriculum-archaeology/CONTEXT.md`

The pipeline is resumable at any lesson. If Stage 02 stalls at lesson 247, it resumes at 247.

---

## For Coursemakers

The pipeline is domain-agnostic. Replace GTM with any practitioner field and the pipeline produces a different course with the same structural properties. See [COURSEMAKER.md](COURSEMAKER.md) — "The Broader Application" section.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution checklist and PR process.

## License

MIT. See [LICENSE](LICENSE).
