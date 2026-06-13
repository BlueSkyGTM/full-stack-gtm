# KV Cache, Flash Attention & Inference Optimization

---

## Beat 1: Hook — The Autoregressive Tax

**Description:** Autoregressive models recompute attention over all prior tokens at every generation step. Demonstrate the quadratic cost with a short script that simulates naive token-by-token generation and prints cumulative FLOPs. The number explodes fast enough to make the problem visceral before any explanation.

---

## Beat 2: Concept — KV Cache Mechanics

**Description:** Explain the mechanism: at each forward pass, the Q matrix is new but K and V for prior positions are identical to the previous step. Caching them trades memory for compute. Walk through the shape transformations — `[batch, seq_len, heads, dim]` — and show where the cache is read/written during attention scoring. Then name the implementation: HuggingFace `generate()` uses KV cache by default; `past_key_values` is the parameter that holds it.

**Exercise hooks:**
- *Easy:* Print the shape of `past_key_values` from a small model before and after one generation step.
- *Hard:* Disable KV cache, generate 50 tokens, measure wall time. Re-enable, repeat. Print the ratio.

---

## Beat 3: Mechanism — Flash Attention and IO-Awareness

**Description:** Standard attention materializes the full `[seq_len, seq_len]` attention matrix in HBM. Flash Attention (Dao et al., 2022) [CITATION NEEDED — concept: Flash Attention paper exact title and year] avoids this by tiling the computation in SRAM and using online softmax — accumulating partial results without ever writing the full matrix back to HBM. Explain the algorithm: block Q, block K/V, compute partial softmax with correction, write only the output. Memory goes from O(N²) to O(N). Then show where this lives in practice: PyTorch 2.0+ `torch.nn.functional.scaled_dot_product_attention` dispatches to Flash Attention kernels when CUDA conditions are met.

**Exercise hooks:**
- *Medium:* Profile memory allocation for `scaled_dot_product_attention` at seq_len 512 vs 4096 with and without Flash Attention backend. Print peak memory delta.

---

## Beat 4: Use It — Inference Cost Modeling for GTM Tooling

**Description:** GTM redirect: any tool in the **AI SDR / Personalization cluster** (Zone 2) that runs batched inference over prospect data pays for latency and compute per token. Model the cost: given a prompt template of length `P`, generation length `G`, and batch size `B`, compute total attention FLOPs with and without KV cache. Print a cost table. This is the mechanism that determines whether a Clay waterfall enriching 10K records with LLM calls is economically viable or requires response caching and shorter prompts. The redirect is real — inference cost directly constrains GTM automation scale.

**Exercise hooks:**
- *Medium:* Write a function that takes `(prompt_tokens, gen_tokens, batch_size)` and prints estimated FLOPs and HBM reads with and without KV cache. Parameterize it for GPT-4-class dimensions.

---

## Beat 5: Ship It — Production Inference Configuration

**Description:** Cover the levers a practitioner actually touches: KV cache quantization (FP8/INT8), `max_batch_size`, `max_tokens`, PagedAttention (vLLM implements virtual memory paging for KV cache to eliminate fragmentation). Show a working vLLM launch command and a Python script that hits the OpenAI-compatible endpoint, then prints per-request latency and throughput. Explain what each flag does to memory budget and concurrency — no marketing, just arithmetic: available GPU memory minus model weights minus KV cache per sequence equals number of concurrent requests.

**Exercise hooks:**
- *Hard:* Launch vLLM with a small model (e.g., `facebook/opt-125m`), send 10 concurrent requests with different `max_tokens`, print the latency distribution and GPU memory usage from the `/metrics` endpoint.

---

## Beat 6: Prove It

**Description:** Five questions. No trivia.

1. Given a seq_len of 2048 and head_dim of 64 with 32 heads, calculate the HBM bytes consumed by the KV cache for a single sequence at FP16. *(tests shape reasoning + memory math)*
2. Explain why Flash Attention does not change the mathematical output of attention — what property does it preserve and how? *(tests mechanism understanding, not memorization)*
3. A GTM team reports that their LLM enrichment pipeline costs $2K/month at 5K records. They want to scale to 50K records. Identify which inference parameter (prompt length, generation length, batch size, KV cache precision) is the highest-leverage dial and why. *(tests cost modeling from Beat 4)*
4. vLLM uses PagedAttention. Describe the problem it solves that naive KV cache allocation creates under concurrent requests. *(tests production knowledge from Beat 5)*
5. Write the arithmetic for "available GPU memory minus model weights minus KV cache per sequence equals number of concurrent requests." Given a 7B parameter model at FP16 on an A100 (80GB), and a max sequence length of 4096, calculate the max concurrent requests. *(tests the ship-it arithmetic)*

---

## Learning Objectives

1. **Calculate** the compute and memory cost of autoregressive generation with and without KV caching.
2. **Explain** the Flash Attention tiling algorithm and why it reduces HBM reads/writes without changing attention output.
3. **Configure** inference server parameters (batch size, KV cache limits, quantization) to hit target latency and throughput.
4. **Model** inference cost for a GTM automation pipeline given prompt length, generation length, and request volume.
5. **Compare** naive, KV-cached, and PagedAttention allocation strategies for concurrent inference workloads.