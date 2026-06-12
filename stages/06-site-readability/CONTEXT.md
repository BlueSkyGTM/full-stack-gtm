<!-- Agent: Lyra-code -->
# Stage 06: Site Readability Overhaul

Implement modular lesson components without breaking the existing render pipeline.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Design system snapshot | `../00-a-curriculum-archaeology/output/design-system-snapshot.md` | Full file | Rendering stack + existing patterns to extend |
| Hybrid lessons | `../02-lesson-injection/output/hybrid-lessons/` | Sample lessons | Before/after reference |
| Render pipeline | `{{REPO_URL}}` | ui.js, catalog.js | Do not break these |
| Lyra code brief | `../00-c-agent-setup/output/agent-briefs/lyra-code-brief.md` | Full file | Site architecture constraints |
| Illustration pipeline | `../../shared/illustration-pipeline.md` | Full file | Tier 1+2 illustration generation for The Concept beats |
| Quality standards | `../../shared/quality-standards.md` | Illustration Quality section | Tier selection logic and validation requirements |

## Prerequisites

**GLM-image swap must be implemented before this stage runs.** `scripts/generate-image.ts` in smart-illustrator must have the Gemini SDK replaced with the GLM-image API call (see illustration-pipeline.md → Tier 1 implementation notes). Stage 06 will fail if this is not done. Confirm by running one test generation call before starting.

## Process

1. Run /design-review on current site — 80-item audit
2. Prioritize: lessons that look thin, unreadable, or not visually significant
3. Confirm Mermaid rendering is wired: the site must parse ` ```mermaid ` fenced blocks and render them as SVG (mermaid.js CDN include or SSG plugin). If not already present, add it. Verify with one test block before proceeding.
4. Implement modular lesson components per the rendering stack from design-system-snapshot — do not break the existing render pipeline
5. Generate Tier 1 (GLM-image) and Tier 2 (Excalidraw) illustrations — per illustration-pipeline.md tier selection logic:
   - Run sequentially, not in parallel (two Playwright instances on the same machine will conflict)
   - Checkpoint each lesson's illustration status to `output/illustrations/progress.json` before moving to the next — resume from checkpoint on failure
   - Do not batch-queue Tier 2 (Excalidraw) — run render-and-inspect per lesson, write output, advance
6. Drop illustration outputs into `output/illustrations/{{PHASE}}/{{LESSON}}/` with alt-text as specified in illustration-pipeline.md
7. Validate all illustrations and site components in browser before /review gate

## Pacing, Revelation, and Loop Design — PRIMARY OUTPUT

**This is not a visual polish stage.** The readability overhaul is the last moment to set the rhythm of the student experience at the structural level. Once Stage 02 lessons are injected and Stage 08 agents are wired, retrofitting pacing means touching hundreds of files. Get it right here.

### Three Core Loops

The curriculum must maintain three loops simultaneously — neglect any one and students drop at the hard parts:

**Micro loop (every session):** read docs → run code exercise → pass quiz. Friction must be low enough to do tired. One quiz question at a time. Terminal exercise, no per-session setup. If a lesson feels like three lessons, it is too long — flag it for Stage 09 quality pass.

**Mid loop (every few phases):** adding a component to the mission command repo. This is the merchant visit — the student fought through five hard lessons and now drops a working scraper into their repo and sees it pull real data. Stage 06 must identify where these moments fall in the phase sequence and ensure the site signals them as rewards, not checkboxes.

**Macro loop (major milestones):** two designed revelations. Both must be written as deliberate beats, not discovered accidentally.

### Revelation 1 — Helix Wakes Up (after Phase 04-05)

The student completes something hard. They open their mission command and Helix is there — ready, context-loaded, knowing their ICP and voice. No banner. No feature announcement. Just present. The aha is quiet: *the repo is alive*. This signals that they have left the tutorial and entered the real system. They can breathe. Do not over-engineer this moment. The architecture earns it — Stage 06 just has to not smother it with fanfare.

**Implementation constraint:** Helix must be introduced gradually across early phases before this reveal. Students should feel the tutor becoming more useful, not encounter it cold. The site's lesson progression must reflect this ramp.

### Revelation 2 — The Loop Reveal (Stage 10 validation beat)

Written into Stage 10, not Stage 06 — but Stage 06 must lay the groundwork. The final reveal: the student realizes the course they just completed was itself a loop. The quiz factory, the manifest, the batch orchestration, the operator kit running Lyra — they were inside a loop the entire time. The course ate its own cooking.

This moment can be spoiled and it does not matter. Knowing it is coming does not diminish it because the student has to earn it first.

**Stage 06 responsibility:** ensure the site does not accidentally surface this framing early. No premature references to the course-as-loop in lesson copy, tooltips, or progress UI. The reveal is a Stage 10 decision — protect it.

### Difficulty Curve — Required Pacing Map Output

Games oscillate: hard area → power spike → steamrolled → power spike earned. The curriculum must do the same. The AI engineering fundamentals (transformers, attention, embeddings) are genuinely hard. The GTM application phases that follow should feel like a power spike — students see exactly where the theory lands in Clay, in signal scoring, in outreach drafting. Then the signal engine work steamrolls them again before the mission command unlock feels earned.

**Stage 06 must produce `output/pacing-map.md`** — a phase-by-phase difficulty rating (hard / medium / power spike) with the rationale. The pacing map is not decoration — it is a pipeline input. On first run, it is consumed by Stage 09 (quality pass), which flags hard-phase clusters and corrects Stage 02 lesson content. On subsequent iterations, Stage 02 reads pacing-map.md before re-injecting lessons. Do not treat pacing-map.md as a first-run Stage 02 prerequisite — Stage 06 runs after Stage 02 and produces the map; Stage 09 closes the loop.

### What Gamification Means Here

Gamification is applying game design principles — pacing, loop structure, difficulty oscillation, earned revelation. It is not game references, character select screens, or Zelda analogies in lesson copy. The site's game theme is aesthetic. The loop design is structural. Do not conflate them.

The rule: if a student can drop in at any phase after a break and feel oriented within one lesson, the micro loop is working. If they occasionally feel overpowered (GTM application clicks, mission command component lands), the mid loop is working. If they finish Stage 10 and feel the weight of what they built, the macro loop worked. Optimize for all three — not for streaks, daily rewards, or engagement metrics.

## Audit

| Check | Pass Condition |
|-------|---------------|
| Render pipeline intact | ui.js and catalog.js behavior unchanged |
| Lessons look significant | No lesson renders as a thin block of text |
| Components are modular | Each component can be updated independently |
| Pacing map produced | `output/pacing-map.md` exists with phase-by-phase difficulty ratings |
| Helix ramp documented | Site progression reflects gradual Helix introduction before Revelation 1 |
| Revelation 2 protected | No premature loop-reveal framing in lesson copy or progress UI |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| `site-components/` | `output/` | Updated lesson HTML components |
| `pacing-map.md` | `output/` | Phase-by-phase difficulty curve — read by Stage 02 |
