# Long-Context Evaluation — NIAH, RULER, LongBench, MRCR

## Hook

Models advertise 128k–1M token context windows. Retrieval and reasoning accuracy degrade with depth and complexity. If you're stuffing account research, enrichment payloads, or multi-document briefs into a single prompt, you need to know *where* it breaks and *how* to measure that — not rely on vendor benchmarks.

## Concept

Four benchmarks, each probing a different failure mode in long-context processing:

- **NIAH (Needle In A Haystack):** Places a target fact at controlled depth positions within padding text. Measures simple retrieval accuracy across a depth × context-length grid. Reveals positional bias — models often retrieve better from start/end ("lost in the middle" effect, [Liu et al. 2023, CITATION NEEDED — concept: lost-in-the-middle positional bias]).

- **RULER:** Extends NIAH from single-fact retrieval to multi-variable tracking, aggregation queries, and multi-needle tasks over synthetic long contexts. Probes whether the model can *reason* over distributed information, not just locate it. Configurable task complexity.

- **LongBench:** Multi-task benchmark (~20 tasks across QA, summarization, few-shot learning, code, dialogue) with real-world long documents up to ~32k tokens. Tests practical downstream performance rather than synthetic retrieval. Includes metrics like F1, ROUGE, exact match depending on task type.

- **MRCR (Multi-hop Reasoning on Context Retrieval):** [CITATION NEEDED — concept: MRCR benchmark origin paper and task specification]. Multi-hop reasoning tasks requiring the model to chain facts scattered across a long context. Probes compositional failure — can the model join A→B→C when each fact sits at a different depth?

Key mechanism across all four: **position-dependent accuracy decay**. The diagnostic question is whether decay is linear, U-shaped (better at edges), or threshold-based (works until it doesn't).

## Demo

Build a minimal NIAH test harness. Given an API endpoint (OpenAI-compatible), insert a needle ("The secret code is XJ-3847") at specified depth percentages within haystack padding. Query the model for the secret code. Sweep depth 0%→100% across multiple context lengths. Print a depth × context-length accuracy grid to stdout.

Exercise hooks:
- **Easy:** Run the provided NIAH sweep at three fixed depths (0%, 50%, 100%) on a 4k context. Print pass/fail per depth.
- **Medium:** Sweep 10 depth positions × 3 context lengths (2k, 4k, 8k). Output a formatted grid showing pass/fail.
- **Hard:** Implement a two-needle variant — insert two independent facts at different depths, query both in one prompt. Measure whether retrieval of one needle correlates with retrieval of the other.

## Use It

When compiling GTM enrichment payloads — account 10-Ks, recent news, tech stack signals,ICP match criteria — into a single prompt for scoring or outreach generation, context depth matters. If your enrichment payload places the ICP match criteria at position 60% in a 30k-token context, and your model has a mid-context retrieval dip, your scoring silently degrades.

GTM cluster redirect: **Zone 02 — Enrichment** and **Zone 03 — Scoring & Qualification**. Long-context evaluation tells you whether you can safely concatenate multiple enrichment sources into one prompt or must retrieve-then-rerank with shorter contexts.

Exercise hooks:
- **Easy:** Take a sample enrichment payload (provided). Identify where the key scoring signals sit by token position. Compare against your model's NIAH profile.
- **Medium:** Structure the same enrichment data so that the highest-priority signals fall in the top and bottom 20% of the context. Re-run scoring and compare output quality.

## Ship It

Produce a reusable evaluation script that:
1. Accepts a model endpoint and context-length configuration.
2. Runs a NIAH sweep (configurable depths and context lengths).
3. Outputs a CSV with columns: `context_length, depth_pct, needle_found, response_latency_ms`.
4. Includes a second mode that runs a multi-needle variant (2 needles, configurable depth separation).

Deliverable: the script plus a one-paragraph interpretation of the output grid for your primary model, noting any depth range where accuracy drops below 80%.

Exercise hooks:
- **Easy:** Run the script against any OpenAI-compatible endpoint. Submit the CSV output.
- **Medium:** Add a `--mode ruler-lite` flag that inserts two needles and asks an aggregation question (e.g., "What is the sum of the two secret numbers?"). Test whether the model retrieves *and* reasons.
- **Hard:** Extend the script with a LongBench-style task: feed a real document (>8k tokens), ask a comprehension question with a known answer, measure exact match. Compare against the synthetic NIAH results for the same context length.

## Review

- Why does NIAH use a depth × context-length grid rather than a single accuracy number?
- What failure mode does RULER expose that NIAH does not?
- A model scores 95% on NIAH at all depths but fails RULER aggregation tasks. What does this tell you about the model's long-context capability?
- You observe a U-shaped accuracy curve on NIAH (high at edges, low in middle). How should you structure a GTM enrichment prompt to mitigate this?
- LongBench uses real documents; NIAH uses synthetic padding. What does each choice trade off?

Exercise hooks:
- **Easy:** Given a sample NIAH output grid (provided), identify the depth range where accuracy drops below 70%.
- **Medium:** Write a 3-sentence interpretation of why a model might pass NIAH but fail a multi-hop benchmark, citing the specific capability gap.
- **Hard:** Propose a fifth benchmark task that would stress-test a model's ability to handle the kind of structured, repetitive data common in GTM enrichment (company lists,ICP criteria, signal feeds). Specify the needle format, haystack structure, and scoring metric.