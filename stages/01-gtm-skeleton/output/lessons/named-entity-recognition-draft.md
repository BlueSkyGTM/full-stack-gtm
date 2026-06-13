# Named Entity Recognition

## Hook It
Every inbound signal — job postings, press releases, LinkedIn bios — arrives as unstructured text. NER is the mechanism that converts that text into structured records: company, person, role, location, date. Without it, you're hand-reading every signal. With it, you feed the waterfall.

## Ground It
NER treats entity extraction as a token-classification problem: each token in a sequence gets a label (PERSON, ORG, GPE, DATE) or O for "outside any entity." The BIO tagging scheme encodes boundaries — B-ORG starts an organization name, I-ORG continues it. Two families of approaches: rule-based matchers (fast, deterministic, blind to context) and statistical models (CRF, transformer-based — slower but sensitive to "Apple" as ORG vs. PROD based on surrounding words). spaCy ships both; the English pipeline `en_core_web_sm` uses a transition-based parser with greedy decoding.

## Show It
Runnable Python script using spaCy's pre-trained pipeline on a job posting excerpt. Prints each entity's text, label, and character offsets. Second example swaps in a `Matcher` pattern to catch entities the statistical model misses — demonstrates the rule-based fallback path.

## Try It
- **Easy:** Run the provided spaCy pipeline on three prospect bios. Print extracted entities grouped by label type.
- **Medium:** Write a `Matcher` pattern that extracts job titles containing "VP" or "Director" from the same bios. Compare Matcher output to the statistical model's ORG/PERSON hits — note what each catches that the other misses.
- **Hard:** Feed five real job postings through the pipeline. Write code that produces a structured JSON record per posting with fields: `company`, `role`, `locations`, `tech_stack`. Evaluate precision on two postings by hand-labeling ground truth.

## Use It
GTM cluster: **Enrichment (Zone 2).** NER is the extraction step that turns raw text signals into the structured attributes Clay's waterfall consumes. A job posting arrives as a string; NER pulls company name and location; those feed into company enrichment lookups downstream. When the statistical model misses domain-specific entities (product names, niche job titles), the rule-based `Matcher` catches them — same fallback pattern the Clay waterfall uses when one provider returns empty.

## Ship It
Production NER pipelines fail on two axes: throughput and confidence. spaCy's small model processes ~10k docs/minute on CPU; the transformer model drops to ~200 but catches more entities. Ship the small model as default, log every low-confidence entity (model probability < 0.7) to a review queue, and patch gaps with `Matcher` patterns until the review queue shrinks. Batch your documents — spaCy's `nlp.pipe()` processes in chunks and avoids per-doc overhead. Track entity extraction rate weekly; if it drops, your source text drifted.

---

**Learning Objectives:**
1. Extract named entities from raw text using a pre-trained spaCy pipeline.
2. Compare statistical and rule-based NER approaches on the same input.
3. Evaluate NER precision and recall against hand-labeled ground truth.
4. Build a `Matcher` pattern to catch entities the statistical model misses.
5. Construct a structured JSON record from unstructured prospect text.

**Exercise Hooks:**
- Easy: extract-and-print from bios
- Medium: dual-approach comparison with Matcher
- Hard: end-to-end extraction with manual precision evaluation