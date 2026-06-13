# Inference Platform Economics — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

## Hook

You're paying for inference right now — probably on the wrong pricing model. This lesson breaks down how six platforms charge for compute, where each model breaks economically, and how to pick based on your actual request pattern, not marketing copy.

## Concept

Introduce the three pricing primitives that compose every inference platform's billing: per-token, per-GPU-second, and per-request. Show how each platform blends these primitives differently. Present the key variables that determine which blend wins: request frequency, payload size, latency tolerance, and cold-start sensitivity.

## Mechanism

Decompose each platform's billing mechanism by inspecting their pricing pages and rate limit documentation. Build a cost model that takes request pattern inputs (QPS, input/output token distribution, concurrency) and outputs projected monthly cost per platform. Identify the crossover points where one platform becomes cheaper than another. Address undocumented billing behaviors (e.g., how cold starts are metered, whether token counting includes formatting tokens) with skeptical defaults.

Platforms covered:
- **Fireworks** — per-token with throughput tiers, batching economics
- **Together** — per-token with dedicated vs shared endpoints
- **Baseten** — per-GPU-minute with model loading overhead
- **Modal** — per-GPU-second with cold start billing granularity
- **Replicate** — per-second billing with per-prediction minimums
- **Anyscale** — per-token with autoscaling latency costs

## Code

Build a CLI cost calculator that accepts a request profile (QPS, avg input tokens, avg output tokens, hours/day) and prints a sorted cost comparison across platforms using their published pricing. Output a markdown table with monthly projections and crossover analysis.

Exercise hooks:
- Easy: Modify the request profile inputs and rerun to compare costs for a batch workload vs real-time workload.
- Medium: Add a new platform's pricing data to the calculator and identify the crossover point where it beats the cheapest existing option.
- Hard: Model cold-start frequency as a function of QPS and autoscaling behavior, then plot how total cost changes as QPS drops below 1 request per minute.

## Use It

GTM enrichment pipelines that run company research, person research, or message personalization at scale are inference-heavy workloads with predictable request patterns. This lesson's cost model directly applies to sizing the inference budget for Zone 2 enrichment and Zone 3 personalization in the GTM topic map — specifically, choosing the right platform for batch enrichment (favoring per-token pricing) vs real-time scoring (favoring per-GPU-second with warm instances).

[CITATION NEEDED — concept: GTM topic map Zone 2/Zone 3 inference cost budgets]

## Ship It

Deliverable: a runnable cost projection script that takes a GTM workload profile (number of accounts, enrichment fields, personalization depth) and outputs a platform recommendation with monthly cost estimate and the assumptions that could invalidate it. Include a one-paragraph writeup explaining which platform you'd pick for three specific GTM scenarios and why.