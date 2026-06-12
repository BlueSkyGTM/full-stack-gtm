# Illustration Pipeline

Three-tier logic for mass illustration generation across 502 lessons. Each lesson in The Concept beat gets exactly one illustration. Tier selection is automatic.

---

## Tier Selection Logic

```
Content type check:
├─ Can be expressed as a sequence, flowchart, or decision tree?
│   └─ YES → Tier 3: Mermaid (inline markdown, zero render dependency)
│
├─ Has spatial components, entities, or system relationships?
│   └─ YES → Tier 2: Excalidraw (architecture diagrams, data flow maps)
│
└─ Abstract concept, metaphor, or analogy that needs visual grounding?
    └─ YES → Tier 1: GLM-image (conceptual illustrations)
```

One tier per lesson. No mixing.

**Boundary case default:** when a concept could fit two tiers (e.g., a pipeline that is both a sequence AND has entities), default to Tier 2 (Excalidraw). It can show both structure and flow.

### Labeled Examples — 5 per tier

**Tier 3 (Mermaid):**
1. `FSRS scheduling intervals` — flowchart: Again → short interval, Good → longer
2. `Lead scoring decision tree` — branching: score > 80? → sequence A / B
3. `API request lifecycle` — sequence diagram: client → enrichment API → CRM
4. `Clay waterfall steps` — sequential flow: Find → Enrich → Transform → Export
5. `Helix modality selection` — decision tree: concept question? → EXPLAIN / stuck? → HINT

**Tier 2 (Excalidraw):**
1. `Vector database architecture` — entities: query → embedding layer → vector store → result
2. `Multi-agent GTM system` — components: signal agent, enrichment agent, copy agent, router
3. `Double Helix course structure` — spatial: AI engineering spine + GTM application layer woven in
4. `CRM as retrieval system` — data flow: account record → embedding → similarity search → context
5. `Enrichment waterfall with retry logic` — distributed system: parallel providers, failure paths, idempotent retries

**Tier 1 (GLM-image):**
1. `Fine-tuning concept` — metaphor: a sculptor refining a rough shape to match a specific mold
2. `Embeddings` — analogy: words as coordinates in a space where meaning = proximity
3. `RLHF` — abstract: human preference arrows nudging a probability distribution
4. `Chain-of-thought prompting` — visual: stepping stones across a river (each step visible)
5. `Overfitting` — metaphor: a tailor making a suit that fits only one person's exact pose

---

## Tier 1: GLM-image (Conceptual)

**Tool:** smart-illustrator (axtonliu/smart-illustrator) with Gemini API swapped for GLM-image API.

**Swap point:** `scripts/generate-image.ts` — replace the Gemini SDK call with GLM-image API endpoint. Input/output contract stays identical: content description in, PNG path out.

**Variable:** `{{GLM_IMAGE_ENDPOINT}}` — set in variable-registry.md, injected at runtime.

**Batch spec:** `references/slides-prompt-example.json` pattern — one entry per lesson needing a Tier 1 illustration. Batch runs with resume support (smart-illustrator has built-in checkpoint).

**Quality gate:** GLM-image outputs are human-reviewed at phase checkpoints (same cadence as Stage 02 phase pauses). No automated render-and-inspect for generated images.

**Output location:** `output/illustrations/{{PHASE}}/{{LESSON}}/concept.png`

**Alt-text:** Every generated illustration must include a corresponding `concept.alt.txt` file — one sentence describing what the image shows, written for a screen reader. Format: `[Tier 1 illustration: description of what the visual depicts]`. GLM-image batch spec generates alt-text alongside each image using the same content description that was passed to the API.

---

## Tier 2: Excalidraw (Architectural)

**Tool:** excalidraw-diagram-skill (coleam00/excalidraw-diagram-skill).

**The differentiator:** render-and-inspect self-validation loop. The agent generates Excalidraw JSON, renders to PNG via Playwright, screenshots the result, and self-corrects layout before writing output. Catches visual layout bugs automatically.

**Usage:** Claude Code skill loaded from `.claude/skills/excalidraw-diagram-skill/`. Run per lesson that needs a Tier 2 diagram.

**Quality gate:** render-and-inspect loop is the gate. Visual output is human-reviewed at phase checkpoints.

**Output location:** `output/illustrations/{{PHASE}}/{{LESSON}}/diagram.svg`

**Alt-text:** Every Excalidraw diagram must include a `diagram.alt.txt` — one sentence naming the components and relationships shown. The render-and-inspect loop generates alt-text as part of the self-correction pass.

---

## Tier 3: Mermaid (Structural)

**Tool:** Mermaid.js — client-side library; parses ` ```mermaid ` fenced blocks and renders SVG in the browser.

**Integration requirement:** mermaid.js must be loaded on the site before any Mermaid diagrams are written. This is a one-line CDN include or SSG plugin — not a codebase clone. Stage 06 confirms and wires this before Tier 3 diagrams go into lesson files.

**Generation:** Claude writes Mermaid syntax directly during Stage 02 lesson injection. No API, no external call, no generation infrastructure. It's text.

**Quality gate:** Stage 06 run-one-test-block check before batch; Stage 10 site render validation catches syntax errors that survived.

**Output location:** inline in lesson file — no separate illustration output.

---

## Integration with the Build Pipeline

| Stage | Illustration action |
|-------|-------------------|
| Stage 02 (Lesson Injection) | Tier 3 (Mermaid) written inline during lesson drafting |
| Stage 06 (Site Readability) | Tier 1 + Tier 2 generated per lesson needing visual grounding; illustration outputs dropped into lesson folders |
| Stage 09 (Quality Pass) | Hypatia flags missing illustrations; Tier 1/2 re-runs for flagged lessons |
| Stage 10 (Validation Run) | Confirm all illustrations render correctly on the live site |

---

## GLM-image API Swap — Implementation Notes

The swap requires three changes to smart-illustrator:

1. `scripts/generate-image.ts` — replace:
   ```ts
   // BEFORE (Gemini)
   const model = genai.getGenerativeModel({ model: "gemini-pro-vision" });
   
   // AFTER (GLM-image)
   const response = await fetch(`${process.env.GLM_IMAGE_ENDPOINT}/v1/images/generations`, {
     method: "POST",
     headers: { "Authorization": `Bearer ${process.env.GLM_API_KEY}`, "Content-Type": "application/json" },
     body: JSON.stringify({ model: "cogview-3", prompt: imagePrompt, n: 1, size: "1024x1024" })
   });
   ```

2. `.env` — add `GLM_IMAGE_ENDPOINT` and `GLM_API_KEY` (already in environment, not committed)

3. `styles/` — adjust style presets if GLM-image produces different aspect ratios than Gemini

Security: `GLM_API_KEY` is a runtime secret — set in environment, never committed. Same handling as `{{GLM_API_KEY}}` in variable-registry.
