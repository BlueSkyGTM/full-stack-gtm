# Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d

## GTM Redirect Rules
This lesson maps to the **AI Infrastructure Cost Optimization** cluster. The redirect: disaggregated prefill/decode cuts per-token cost by bin-packing two fundamentally different compute profiles onto hardware that fits each. This is not a GTM tactic — it is foundational for Zone B practitioners who need to justify inference spend.

---

## 1. Hook — The Dual-Personality Problem

Prefill is compute-heavy (processing all prompt tokens in parallel). Decode is memory-bandwidth-heavy (generating one token at a time from KV cache). Running both on the same GPU means one phase is always starved for its resource while the other sits idle. Disaggregation splits them into separate pools, each right-sized. Outline sentence: practitioner sees why colocated prefill/decode leaves money on the table and why the industry is moving to disaggregated serving.

## 2. Concept — Mechanism: KV Cache Handoff Between Pools

Explain the algorithm: (1) prefill workers process the full prompt and build the KV cache, (2) the KV cache is transmitted over high-bandwidth interconnect to a decode worker, (3) the decode worker generates tokens one at a time using that cache. Cover the scheduling problem — which prefill worker pairs with which decode worker — and the memory vs. compute tradeoff that dictates pool sizing. Then name the implementations: NVIDIA Dynamo (the inference runtime that orchestrates this handoff) and llm-d (the open-source disaggregated serving framework). Explain what problem each solves, how it solves it, and how to configure it.

## 3. Demo — Observable Behavior

Exercise hooks:
- **Easy**: Configure a local two-process simulation where process A builds a mock KV cache tensor and sends it over a Unix socket to process B, which reads it and prints cache shape + transfer latency. Observable: printed shape and microsecond count.
- **Medium**: Run a single-GPU inference baseline, capture time-per-token for prefill vs. decode phases using `transformers` generate with timing hooks. Print the ratio. Observable: prefill ms vs. decode ms, confirming the asymmetric profile.
- **Hard**: Deploy a minimal two-pool setup using llm-d or Dynamo on two GPUs, send a request, and log which GPU handled prefill vs. decode. Observable: routing logs showing pool assignment and KV cache transfer time.

## 4. Use It — GTM Application: Inference Cost Modeling for AI Products

This is foundational for Zone B — practitioners building AI products that serve LLM output at scale. The mechanism maps directly to unit economics: prefill pool cost (compute-optimized instances) + decode pool cost (memory-optimized instances) vs. monolithic pool cost. Exercise hook: given throughput targets and hardware specs, calculate the break-even point where disaggregation is cheaper. Observable: a printed cost comparison table. Redirect: this is the infrastructure reasoning behind "AI Infrastructure Cost Optimization" — not a GTM tactic itself, but the mechanism that makes the cost pitch defensible.

## 5. Ship It — Production Deployment: Pool Sizing, Routing, and Failure Modes

Cover the operational concerns: how to size prefill vs. decode pools (depends on prompt length distribution and target throughput), how request routing works when a decode worker is saturated, what happens when the KV cache transfer becomes the bottleneck (interconnect bandwidth saturation), and how to monitor for the "stranded prefill" problem where prefill workers finish fast but have no available decode workers. Exercise hook: configure a pool-sizing calculator that takes prompt length distribution and outputs recommended prefill:decode ratio. Observable: printed ratio with per-pool utilization estimates.

## 6. Review — Checkpoint and Connections

Summarize: disaggregation exploits the fact that prefill and decode have opposing resource profiles. The cost of KV cache transfer is the tax — it only pays off when the asymmetry savings exceed the transfer overhead. Connect forward to: speculative decoding (another decode-phase optimization), continuous batching (scheduling across disaggregated pools), and KV cache compression (reducing the transfer tax). Exercise hook: write a one-paragraph decision rule for when to use disaggregated vs. monolithic serving, given a workload profile. Observable: printed decision and reasoning trace.