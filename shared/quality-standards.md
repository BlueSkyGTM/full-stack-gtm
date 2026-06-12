# Quality Standards

Master quality reference for the Full-Stack GTM course. Four dimensions: lesson reading, AI interaction, illustrations, curriculum accuracy.

---

## 1. Lesson Reading Quality

Adapted from CLEAR text complexity framework (Achievement Network / Student Achievement Partners).

| Dimension | Standard |
|-----------|----------|
| Sentence length | Max 25 words average per paragraph; no sentence > 40 words |
| Vocabulary load | Max 2 unfamiliar terms introduced per Beat; each defined on first use |
| Cognitive demand | Each Beat has one primary concept; compound concepts get their own Beat |
| Coherence | Each paragraph leads with a claim; supporting detail follows |
| Reading level | Target: professional with no prior AI background can follow The Concept; Build It assumes the target reader |

**Judge prompt:** see `quality-evaluator/lesson-judge-prompts.md` → `CLARITY_JUDGE`

---

## 2. AI Interaction Quality (Helix)

Adapted from EducationQ's five pedagogical dimensions, tuned for Helix's governed-maze architecture.

| Dimension | Standard |
|-----------|----------|
| Responsiveness | Helix identifies the student's current conceptual state before responding — never jumps to answer without diagnosing |
| Scaffolding | Helix starts with the student's existing knowledge; never assumes blank slate |
| Explanation depth | Helix explanations are grounded in the AI concept; GTM applications are derived, not asserted |
| Feedback precision | Incorrect answers receive a targeted correction, not a re-explanation of the whole concept |
| Progression fidelity | FSRS scheduling governs review cadence; Helix does not override scheduled intervals without explicit student request |

**Test harness:** see `vault/helix-test-harness.md` for the teacher→student→evaluator loop used to validate system prompt changes.

---

## 3. Illustration Quality

Three-tier logic. Every lesson in The Concept beat gets exactly one illustration. Tier selection is automatic based on content type.

| Tier | Tool | When to use | Output |
|------|------|-------------|--------|
| 1 — Conceptual | GLM-image (via smart-illustrator) | Abstract concepts, metaphors, analogies | PNG embedded in lesson |
| 2 — Architectural | Excalidraw (via excalidraw-diagram-skill) | System diagrams, data flows, component relationships | SVG/PNG with self-validation loop |
| 3 — Structural | Mermaid | Sequences, decision trees, flowcharts | Inline markdown rendered by site |

**Tier selection rule:** if the concept can be expressed as a graph or flowchart → Mermaid. If it has spatial components, entities, or relationships → Excalidraw. Otherwise → GLM-image.

**Validation:** all Excalidraw and Mermaid outputs pass the render-and-inspect loop (agent renders, screenshots, self-corrects layout) before writing. GLM-image outputs are human-reviewed at phase checkpoints.

**Full spec:** see `illustration-pipeline.md`

---

## 4. Curriculum Accuracy

| Layer | Standard | Owner |
|-------|----------|-------|
| AI engineering content | Matches Made With ML source material | Echo (archaeology agent) at Stage 09 |
| GTM content | Every claim has a citation in the lesson's `## Sources` block | Hypatia at Stage 09 |
| Double Helix weave | GTM strand names the specific AI concept in its first sentence | Stage 02 audit + Stage 09 |
| Voice consistency | All lessons match helix-voice.md benchmark | Hypatia voice audit at Stage 09 |

**Rubric:** `quality-evaluator/lesson-judge-prompts.md` covers accuracy, weave quality, and format compliance.

---

## Quick Reference

| What breaks quality | Where it gets caught |
|---|---|
| Vague GTM claim | Stage 02 audit (weave-not-parallel check) |
| Missing citation | Stage 02 audit (## Sources required) |
| Voice drift | Stage 09 Hypatia voice audit |
| Thin lesson beat | Stage 09 alignment score |
| Broken illustration | Render-and-inspect loop at generation time |
| Reading level too high | Stage 09 CLARITY_JUDGE prompt |
| Helix bad pedagogy | helix-test-harness.md run before Stage 05 ships |
