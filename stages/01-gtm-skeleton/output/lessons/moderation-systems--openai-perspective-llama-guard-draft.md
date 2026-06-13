# Moderation Systems — OpenAI, Perspective, Llama Guard

## Learning Objectives

1. Implement multi-provider content moderation checks and parse structured category responses.
2. Compare toxicity scoring outputs across OpenAI Moderation, Perspective API, and Llama Guard.
3. Configure category-specific blocking rules with tunable thresholds for different GTM content types.
4. Build a moderation waterfall that tries providers in sequence and stops on the first definitive result.
5. Evaluate false positive and false negative rates against a labeled test corpus.

---

## Beat 1: Hook — The Unfiltered Pipeline

You just scraped 10,000 company descriptions and fed them into an LLM to generate personalized outreach. Three of them produced outputs that would get your domain flagged as spam—or worse. Moderation is the gate you didn't know you needed until it was too late.

---

## Beat 2: Concept — Classification at the Content Gate

Content moderation is a multi-label classification problem: each input can trigger zero or more harm categories simultaneously. Three architectures dominate production use: hosted classifier APIs (OpenAI), regression-scored toxicity models (Perspective), and instruction-tuned guard models (Llama Guard). Each produces a different signal shape—boolean flags, float scores, or natural-language safety assessments—and each covers a different slice of the harm taxonomy.

---

## Beat 3: Mechanism — How Each Provider Classifies

**OpenAI Moderation API** — A multi-label classifier that evaluates input against 11 categories (hate, sexual, violence, self-harm, etc.) and returns both boolean flags and per-category confidence scores between 0 and 1. Stateless, single-pass, no context window.

**Perspective API (Jigsaw/Google)** — A regression model trained on human-annotated comments (originally Civil Comments dataset) that returns toxicity probability scores. You request specific attributes (TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, etc.) and get floats back. Scores are calibrated against human rater agreement, not binary labels.

**Llama Guard (Meta)** — An instruction-tuned Llama model that accepts a prompt + response pair (or just a single text), classifies them against a 6-category safety taxonomy, and returns `safe` or `unsafe` with a violated category name. Open-weight, runs locally, requires a prompt template that encodes the taxonomy.

Key mechanism difference: OpenAI and Perspective are fixed-classifier APIs—deterministic given the same input. Llama Guard is a generative model prompted to classify, which means it can refuse its own classification or produce inconsistent formatting if the prompt shifts.

---

## Beat 4: Use It — Moderation as a GTM Enrichment Filter

In GTM pipelines, moderation sits at two enforcement points: **inbound filtering** (scraped company data, user-submitted content, enriched account descriptions) and **outbound validation** (AI-generated emails, subject lines, LinkedIn messages). This maps directly to the **Enrichment → zone** in the GTM topic map—specifically the quality gate that runs after data enrichment and before any AI-generated content reaches a prospect.

A moderation waterfall for GTM content: (1) check OpenAI Moderation first for speed and category coverage, (2) if any score lands in a gray zone between 0.3–0.7, escalate to Llama Guard for context-aware reassessment, (3) use Perspective as a continuous toxicity scorer for lead-scoring signals on user-generated content.

[CITATION NEEDED — concept: GTM enrichment quality gates mapping to specific pipeline stages]

**Exercise hooks:**
- *Easy*: Write a script that sends a test string to the OpenAI Moderation API and prints all category scores above 0.1.
- *Medium*: Build a function that accepts a text input, runs it through OpenAI Moderation, and returns a pass/fail decision based on configurable per-category thresholds loaded from a JSON file.
- *Hard*: Implement a two-stage moderation waterfall: OpenAI Moderation as the first gate, Llama Guard as the escalation path for gray-zone scores, with structured logging of every decision.

---

## Beat 5: Ship It — Production Moderation Middleware

A production moderation system needs: (1) a unified interface that normalizes outputs from all three providers into a single schema, (2) configurable routing rules that specify which providers to call and in what order, (3) threshold configuration per content type (e.g., subject lines get a stricter violence threshold than company descriptions), and (4) a results store for threshold calibration over time.

The deployable artifact is a moderation service that accepts text + content_type, runs the configured waterfall, returns a structured `{ safe: bool, flags: [], scores: {}, provider_used: str }` response, and logs the full result for later evaluation.

**Exercise hooks:**
- *Easy*: Write a Python class that wraps OpenAI Moderation API calls and returns a standardized `{ safe, flags, scores }` dict regardless of raw API shape.
- *Medium*: Extend the class to support a fallback chain: try OpenAI first, if the API returns an error or times out, fall back to Perspective API, log which provider was used.
- *Hard*: Build a complete moderation service with per-content-type threshold configs loaded from a YAML file, provider fallback chains, structured JSON logging of every moderation decision, and a CLI interface that processes a CSV of texts and outputs a CSV with moderation results appended.

---

## Beat 6: Evaluate — Measuring What You Actually Block

Moderation systems have two failure modes that matter in production: **false positives** block legitimate GTM content (a company description mentioning "killing it in Q4" flagged as violence), and **false negatives** let harmful content through. You measure both by running a labeled test corpus through your configured pipeline and computing precision, recall, and F1 per category.

The labeled corpus comes from two sources: (1) synthetic test cases you write yourself for your specific GTM domain (financial services language, SaaS jargon, sales metaphors that trigger false positives), and (2) public benchmark datasets like the ToxiGen dataset or OpenAI's own moderation eval sets.

Latency matters too. OpenAI Moderation returns in ~100ms. Perspective in ~200ms. Llama Guard (local inference) in ~1–3s depending on hardware. A waterfall that escalates to Llama Guard on 15% of inputs has a p99 latency driven by that slow path.

**Exercise hooks:**
- *Easy*: Write a test script that feeds 10 known-safe GTM phrases and 10 known-unsafe phrases through your moderation function and prints a 2×2 confusion matrix.
- *Medium*: Build an evaluation harness that loads a JSONL test corpus with `text` and `expected_safe` fields, runs each entry through your moderation pipeline, and outputs per-category precision/recall/F1 along with average latency per provider.
- *Hard*: Generate a domain-specific test corpus of 100 GTM phrases (50 safe sales language, 50 edge cases), run a threshold sweep across 5 different threshold configurations, plot false positive rate vs. false negative rate for each config, and recommend the optimal threshold for a GTM enrichment pipeline where false positives cost more than false negatives.

---

## GTM Redirect Rules

- **Use It** redirect: This is the enrichment quality gate in Zone [CITATION NEEDED — concept: enrichment zone mapping in GTM pipeline stages]. Moderation filters scraped and AI-generated content before it enters your CRM or reaches a prospect.
- **Ship It** redirect: This is the moderation middleware that plugs into any GTM data pipeline—enrichment workflows, AI email generation, lead scoring feature extraction.
- **Foundational note**: Content moderation is foundational for any GTM system that processes user-generated text or generates outbound content with LLMs. The specific GTM application depends on where untrusted text enters your pipeline.