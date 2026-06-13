# Paper Writer

## Hook

You need to produce a 3,000-word technical paper by Friday. The AI can generate it — but raw generation produces shallow, structureless text that reads like a padded summary. This lesson covers the decomposition pattern: how to break a document into outline, section prompts, and assembly so the output reads like a human wrote it with a plan.

---

## Concept

**Mechanism: Outline-then-fill decomposition.** A single prompt asking for a full paper produces low-quality output because the model has no structural scaffold to allocate tokens against. Instead: generate an outline first, validate the outline, then fill each section independently. This is the same pattern as chain-of-thought — externalize the plan before executing.

Three components:
1. **Outline generation** — produce a structured heading tree with word-count budgets per section
2. **Section-by-section drafting** — each section gets its own prompt with context (previous section summary, target audience, tone)
3. **Assembly and smoothing** — concatenate sections, then run a final pass for transitions and consistency

Token budget discipline: a 3,000-word paper at ~1.3 tokens/word needs ~4,000 output tokens. If you request it in one shot, the model compresses. Split into 6 sections at ~500 words each, and each section gets room to develop.

---

## Demo

Build a Python script that:
1. Accepts a topic and target word count
2. Calls the Claude API to generate an outline with section budgets
3. Loops through each section, calling the API with the outline + previous section summary
4. Assembles the final document
5. Prints total word count and section-by-section word counts to confirm budget adherence

Observable output: the assembled paper and a word-count verification table printed to terminal.

---

## Use It

**GTM Redirect: Sales enablement and competitive intelligence papers.**

[CITATION NEEDED — concept: GTM content cluster mapping for long-form AI-generated assets]

This decomposition pattern maps directly to producing:
- Competitive battle cards (outline = competitor axes, sections = per-competitor analysis)
- Vertical-specific one-pagers for sales (outline = industry pain points, sections = per-vertical value props)
- Account research dossiers (outline = company profile dimensions, sections = research per dimension)

The mechanism is the same: structured outline → section-level generation → assembly. The GTM application differs only in the outline schema and tone instructions per section.

---

## Ship It

**Easy:** Modify the demo script to accept a custom outline schema (JSON file) instead of generating one. Test with a 3-section schema. Print the assembled output.

**Medium:** Add a "review pass" step — after assembly, send the full paper back to the model with the instruction "identify any contradictions, unsupported claims, or sections that break the stated tone." Print the review notes alongside the paper.

**Hard:** Build a pipeline that generates a competitive battle card from a list of competitor URLs. Scrape each URL for key claims, feed the claims into the outline-then-fill pipeline, and output a formatted markdown battle card with a source attribution section. Print the card and the list of sources used.

---

## Evaluate

Five quiz questions grounded in the lesson's mechanisms:

1. **Mechanism identification:** Why does a single-prompt paper generation produce lower quality than outline-then-fill? (Tests: token budget compression vs. structured allocation)

2. **Component function:** What is the purpose of passing the previous section's summary into the next section's prompt? (Tests: context continuity mechanism)

3. **Token math:** A 4,500-word paper is split into 9 sections. At 1.3 tokens/word, approximately how many output tokens should each section prompt request? (Tests: budget calculation)

4. **Failure mode:** A generated paper has consistent tone but repeats the same example in three different sections. Which part of the decomposition pattern likely failed? (Tests: section isolation vs. cross-section deduplication)

5. **GTM application:** You need to generate 20 vertical-specific one-pagers. Which part of the pipeline changes per vertical — the outline schema, the section prompts, or the assembly step? (Tests: parameterization of the decomposition pattern)

---

## GTM Redirect Rules (reference)

- **Cluster:** Content engine / scalable content production
- **Redirect appears in:** Use It, Ship It
- **Specificity:** "outline-then-fill decomposition for battle cards" not "useful for content"
- **If no clean map exists:** foundational for Zone 2 (Inbound)