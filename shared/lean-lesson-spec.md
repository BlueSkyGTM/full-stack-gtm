# Lean Lesson Spec — Loop 1 (Fundamentals)

The reconciliation target and the generation target for Loop 1. The GLM-5.2
Taskmaster judges every handler output against this spec (it lives in the 5.2
system context, not in the handler prompt — handlers stay on the governed-maze
extract).

**Target length:** 1,200–1,800 words. Lean is the point: momentum, not endurance.

---

## Required sections (in order)

1. `# <Title>`
2. `## Learning Objectives` — exactly 3, each starting with a verb (Implement / Trace / Diagnose / Compare / Build / Configure). No passive verbs ("understand", "learn about").
3. `## The Problem` — the hook. One real failure or stakes moment in the first two sentences (a deal lost, a model misclassifying, a pipeline breaking). Not a section-header cold open.
4. `## The Concept` — the AI mechanism, declarative prose, one claim per paragraph. Exactly one diagram (Mermaid fenced block, valid syntax).
5. `## Build It` — runnable, from scratch. Real code (Python/PyTorch/stdlib), prints observable output. This is the non-tourist skill; it stays.
6. `## Use It` — the GTM weave. A **small runnable slice** (15–30 lines) that fires what Build It just produced against one real GTM task (Clay enrichment, ICP scoring, outreach draft, signal classification). The AI mechanism is **named in the first sentence**. Prose-only weaves are rejected — this is the moat, it must run.
7. `## Exercises` — exactly 2, light. One reinforces Build It, one extends the weave.
8. `## Key Terms` — 5–8 terms, one line each.
9. `## Sources` — real citations (papers with authors/year/arXiv where applicable). **Zero `[CITATION NEEDED]` markers** for ship-ready.

**Not in Loop 1:** `## Ship It` (production deployment), 5 graduated exercises, full multi-tool GTM pipelines. Those are harvested to `intermediate-seed/` and become Loop 2.

---

## Two-tier gate (what the GLM-5.2 judge enforces)

**structure-complete** (blocks the write — a lesson missing these is truncated/malformed):
- Balanced code fences (even count of ```)
- All 9 required `^## ` headers present, in order, none duplicated
- Each section has real content (≥ 200 chars for Concept/Build It/Use It; ≥ 80 for others)
- One valid Mermaid block in The Concept
- `finish_reason == "stop"` on the generating call (never "length")

**ship-ready** (blocks publish — structurally complete but not yet trustworthy):
- Zero `[CITATION NEEDED]` markers
- Use It's first sentence names the specific AI mechanism (not generic "this technique")
- Use It contains a runnable code block, not just prose
- Sources lists real, checkable references

A lesson that is structure-complete but not ship-ready is `done-structure`, not `done`. Only ship-ready lessons publish to the site.

---

## Voice (carried from `shared/quality-standards.md`)

- Mechanism first, tool second. Terminal only, no IDE ever.
- No preamble, no excitement, no marketing claims.
- Sentence length ≤ 25 words average; no sentence > 40 words.
- Max 2 unfamiliar terms per section, each defined on first use.

---

## Bucket-specific generation (what the orchestrator tells the handler)

| Bucket | Handler instruction |
|--------|--------------------|
| `lean-ready` | No generation. Already meets the spec (Ship It is harvested separately). |
| `close` | Front half (Concept + Build It) is good — preserve it verbatim. Finish from where it was cut: complete Use It as a runnable weave, add Exercises / Key Terms / Sources. Do not rewrite what exists. |
| `complete` | Build It was cut mid-section — complete Build It, then write Use It / Exercises / Key Terms / Sources to spec. Preserve Concept + the good part of Build It. |
| `regen` | Write the full lean lesson from the outline + GTM cluster, to this spec. |
