# Text Summarization

## Hook

You've got a 4,000-word earnings call transcript and need a 3-bullet brief for an AE walking into a meeting in 10 minutes. Text summarization is the mechanism that makes that compression reliable — and the reason it sometimes hallucinates key details.

## Concept

Two distinct algorithms: extractive (select and stitch existing sentences) vs. abstractive (generate new text that captures meaning). Production summarization uses abstractive models — typically transformer-based — which means you trade deterministic output for fluency and risk. Context window limits force a chunking strategy; the map-reduce pattern handles documents longer than the model's token limit by summarizing sections independently, then summarizing the summaries.

## Use It

**GTM Redirect:** Account intelligence enrichment (Zone 01 — ICP & Account Research cluster).

When you pull 10-K filings, press releases, and LinkedIn activity for an account, you need to compress thousands of tokens into a signal brief an AE can scan in 30 seconds. The map-reduce summarization pattern is what Clay's enrichment waterfall uses to condense multi-source firmographic data into structured account notes. If the source documents exceed the model's context window, you chunk first — and every chunk boundary is a potential coherence loss point.

## Code It

**Mechanism:** Token-bounded chunking → sequential summarization → reduce step.

**Exercises:**
- *Easy:* Summarize a single short document using the Anthropic API with a fixed max_token limit. Print the summary and the original token count side-by-side.
- *Medium:* Implement a map-reduce summarizer that splits a long document into overlapping chunks, summarizes each, then summarizes the summaries. Print each intermediate summary and the final result.
- *Hard:* Build a summarizer that detects when the model dropped a named entity present in the source text. Run it against 5 test documents and print a precision score.

## Ship It

**GTM Redirect:** Call intelligence and account research pipelines (Zone 01).

Production summarization fails when the input is messy — transcripts with speaker labels, HTML-stripped filings with broken sentences, or concatenated multi-source data with conflicting information. You need pre-processing to normalize input, a token counter to decide chunk boundaries, and a post-processing pass to enforce output format (bullet count, max length, required entities). The summarization endpoint is cheap; the engineering around it is where time goes.

**Deployment checklist items:**
- Token counting before the API call to prevent truncation
- Overlap strategy for chunk boundaries to avoid losing cross-paragraph references
- Output format enforcement via structured prompts or tool use
- Latency budget: map-reduce is N+1 sequential calls — calculate worst case

## Quiz

5 questions sourced from `docs/en.md` objectives. Questions cover: extractive vs. abstractive mechanism distinction, map-reduce chunking tradeoffs, token limit behavior, named entity preservation failure modes, and output format enforcement patterns.

---

## Learning Objectives (for `docs/en.md`)

1. **Compare** extractive and abstractive summarization mechanisms and identify when each is appropriate.
2. **Implement** a map-reduce summarization pipeline that handles documents exceeding a model's context window.
3. **Evaluate** summary quality by detecting named entity omissions against the source text.
4. **Configure** chunking strategies with controlled overlap to preserve cross-boundary references.
5. **Diagnose** common summarization failure modes — hallucination, truncation, format drift — and apply targeted mitigations.