# T5 & BART — Encoder-Decoder Models

---

## Beat 1: Hook It

The encoder-decoder architecture solves a specific problem: sequences of different lengths and structures going in, different sequences coming out. Single-stack models compress everything into one representation space. Encoder-decoder splits that workload across two specialized stacks joined by cross-attention. This beat opens with a concrete demonstration: feed a messy prospect research note in, get a structured summary out — and trace why a single encoder can't do this well.

---

## Beat 2: Ground It

Cover the mechanism: encoder produces a matrix of contextualized representations, decoder attends to that matrix via cross-attention while auto-regressively generating tokens. Then introduce the two models. T5 frames every NLP task as text-in → text-out using span corruption during pretraining. BART uses a denoising autoencoder approach — corrupt the input (token masking, sentence shuffling, document rotation), reconstruct the original. Compare their pretraining objectives and how those objectives bias downstream performance: T5 toward structured transformation tasks, BART toward generation fidelity.

---

## Beat 3: Map It

Diagram the data flow: input tokens → encoder self-attention → encoder feed-forward → encoder output matrix. Then: decoder self-attention (causal) → cross-attention (query from decoder, key/value from encoder output) → feed-forward → output token. Show the shape transformations at each step. Highlight where cross-attention sits — that single mechanism is the bridge between "understanding" and "generating."

**Exercise hooks:**
- *Easy:* Trace tensor shapes through a minimal 2-layer encoder-decoder forward pass.
- *Medium:* Implement cross-attention from scratch (queries from decoder, keys/values from encoder) and verify output shapes.
- *Hard:* Compare generation quality when cross-attention is ablated vs. decoder-only self-attention on a summarization task.

---

## Beat 4: Build It

Load T5-small and BART-base from Hugging Face. Run a summarization pipeline on real input text — a sales call transcript or ICP research note. Tokenize, encode, decode. Print the encoder's last hidden state shape, the decoder's output logits shape, and the generated tokens at each step using `output_scores=True`. Show the mechanism in motion.

**Exercise hooks:**
- *Easy:* Run T5-small summarization on three inputs of different lengths; print and compare encoder hidden state shapes.
- *Medium:* Extract cross-attention weights from a BART forward pass and visualize which input tokens the decoder attended to at each generation step.
- *Hard:* Fine-tune T5-small on a dataset of 50 input→output pairs (e.g., raw notes → structured summaries) and evaluate generation quality before and after.

---

## Beat 5: Use It

Encoder-decoder models map to GTM Zone 1 (ICP & Enrichment) — specifically, transforming unstructured research into structured account data. The canonical application: take a long-form prospect research dump, generate a condensed account brief. This is the "summarization-as-enrichment" pattern. T5's text-to-text framing also handles format transformation: free-text company description → structured fields. This is not about replacing Clay waterfalls — it's about preprocessing messy inputs before they hit your enrichment pipeline.

**GTM Cluster:** Zone 1 — ICP & Enrichment, enrichment data transformation layer.

---

## Beat 6: Ship It

Deploy a T5-small summarization endpoint that accepts raw text and returns a condensed summary. Wrap it in a FastAPI endpoint. Hit it with a curl command containing a real prospect research note. Print the input length, output length, and latency. That's the MVP. From here, the practitioner can wire it into any enrichment flow that needs text condensation before downstream processing.

**GTM Cluster:** Zone 1 — ICP & Enrichment. This is a preprocessing step for the Clay enrichment waterfall: compress raw research before feeding it to enrichment columns.

---

## Learning Objectives (Draft)

1. **Explain** the role of cross-attention in bridging encoder representations to decoder generation.
2. **Compare** T5's span corruption pretraining objective against BART's denoising autoencoder objective.
3. **Implement** a working summarization pipeline using T5 or BART via Hugging Face transformers.
4. **Extract and inspect** cross-attention weights from a forward pass to verify which input tokens influence generation.
5. **Deploy** a minimal encoder-decoder inference endpoint and measure latency on real-length inputs.