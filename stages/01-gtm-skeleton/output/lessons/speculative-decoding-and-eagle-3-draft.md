# Speculative Decoding and EAGLE-3

## GTM Redirect Rules

Speculative decoding is an inference-time acceleration technique. It does not have a direct GTM workflow mapping. The redirect is **foundational for Zone C (Enrich)** — any pipeline that runs batch LLM inference over thousands of records (company classification, persona extraction, signal scoring) benefits directly from lower latency and higher throughput per GPU. The mechanism is infrastructure cost and speed, not a GTM workflow step.

---

## Beat 1: Hook

**Why inference latency is a compounding bottleneck.** When you run 50k records through an enrichment pipeline and each LLM call takes 800ms, you're waiting ~11 hours. Speculative decoding cuts that wall-clock time without changing model weights or buying more GPUs. Frame the problem in concrete pipeline terms — throughput, cost-per-inference, and the arithmetic of autoregressive generation's sequential bottleneck.

---

## Beat 2: Concept

**Autoregressive decoding is inherently sequential.** Each token conditions on all prior tokens, so you generate one token per forward pass. Speculative decoding breaks this by introducing a two-model architecture: a small draft model proposes K tokens, then the target model verifies all K in a single forward pass using a parallel scoring trick. Cover the acceptance criterion (the modified rejection sampling from the original Leviathan et al. paper [CITATION NEEDED — concept: speculative decoding original paper, Leviathan et al. 2023]) and why this preserves the exact output distribution of the target model.

---

## Beat 3: Mechanism

**How EAGLE-3 changes the draft strategy.** Traditional speculative decoding uses a separate smaller model as the drafter. EAGLE-3 (draft model uses the target model's own intermediate features — specifically the top-K hidden states from the second-to-last layer — to predict the next token's feature representation, then maps that to a token via the existing LM head) avoids training and maintaining a separate draft model. Walk through: (1) what "feature-level drafting" means vs. token-level drafting, (2) how the drafting head is trained — a lightweight MLP on top of frozen target-model features, (3) the tree-structured speculation where multiple draft paths are evaluated in one target forward pass, and (4) why the acceptance rate is higher than independent-draft approaches (the draft model has access to the same representation space as the verifier).

Key equations to surface: the acceptance probability formula, and why speculation length K trades off against acceptance rate (higher K = more rejections but bigger payoff on acceptance).

---

## Beat 4: Code

**Measure the wall-clock difference.** Two exercises:

- **Easy:** A minimal speculative decoding simulation in pure Python. Two "models" (one fast small vocabulary sampler, one slow correct sampler). Run 100 token generations with and without speculation. Print wall-clock time and acceptance rate. No GPU required — this is a timing and logic demonstration using `time.perf_counter()` and `random.choice` with controlled distributions.

- **Medium:** Profile real inference throughput. Use `vLLM` (which implements speculative decoding) to run a target model with and without a draft model. Measure tokens/second and acceptance rate from the vLLM metrics endpoint. Print a comparison table. Requires GPU access — note this explicitly.

---

## Beat 5: Use It

**Where this matters in a GTM stack.** Any enrichment pipeline that runs thousands of LLM calls in sequence (company classification, ICP scoring, email personalization) hits the throughput ceiling. Speculative decoding is a deploy-time lever: you swap your inference backend config, not your prompt or model. Map this to Zone C (Enrich) — specifically, the difference between "run 10k records through GPT-4-class model at 30 tok/s" vs. "same model at 60 tok/s via speculation" is the difference between a 4-hour batch job and a 2-hour batch job. The GTM redirect: **this is foundational for Zone C** — it is an infrastructure optimization that makes high-volume enrichment economically viable, not a GTM workflow step itself.

Exercise hook: Calculate the cost and time break-even point for enabling speculative decoding on a hypothetical enrichment pipeline (given: tokens per record, number of records, GPU hourly cost, acceptance rate). Print the result.

---

## Beat 6: Ship It

**Configuration checklist and acceptance rate monitoring.** Speculative decoding's value depends entirely on acceptance rate — if the draft model and target model diverge, you lose more time than you gain (each rejection wastes a draft forward pass). Ship a monitoring script: given vLLM's `/metrics` Prometheus endpoint, poll acceptance rate and alert if it drops below a threshold. Print a single-line status output.

Key takeaway: speculative decoding is not a model change, it's a systems change. You deploy it at the inference server level, monitor acceptance rate, and tune the draft model or speculation length K accordingly. EAGLE-3 specifically reduces operational complexity by eliminating the separate draft model — you train a lightweight head once and attach it to the target model.

---

## Learning Objectives

1. **Calculate** the theoretical speedup of speculative decoding given acceptance rate `α`, speculation length `K`, and target/draft forward pass latency.
2. **Implement** a minimal speculative decoding loop (draft → verify → accept/reject) and print acceptance rate and wall-clock time.
3. **Explain** why feature-level drafting (EAGLE-3) achieves higher acceptance rates than token-level drafting with an independent small model.
4. **Configure** speculative decoding in vLLM and extract throughput metrics from the `/metrics` endpoint.
5. **Evaluate** whether speculative decoding is worthwhile for a given enrichment pipeline by computing the break-even acceptance rate from cost and latency constraints.