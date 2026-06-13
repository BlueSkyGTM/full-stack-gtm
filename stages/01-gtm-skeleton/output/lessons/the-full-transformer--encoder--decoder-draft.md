# The Full Transformer — Encoder + Decoder

## Hook

The encoder-decoder architecture is the original Transformer from Vaswani et al. (2017). It was built for translation: encode a source sentence into contextual representations, then decode those representations into a target sentence, one token at a time. Most production LLMs today use *only* the decoder (GPT) or *only* the encoder (BERT-family), but cross-attention — the mechanism that connects encoder output to decoder state — shows up everywhere: retrieval-augmented generation, multimodal models, and any system where one modality conditions another.

---

## Concept

Define the three components that distinguish a full Transformer from a decoder-only stack:

1. **Encoder stack** — bidirectional self-attention over the input sequence. Every token attends to every other token in both directions. Produces a set of contextualized vectors (one per input token).

2. **Decoder stack** — three sub-layers per block instead of two: masked self-attention (causal), cross-attention (queries from decoder, keys/values from encoder output), and a position-wise feed-forward network.

3. **Cross-attention** — the mechanism that bridges the two stacks. Decoder tokens query the encoder's final-layer representations. This is how the decoder "looks back" at source information while generating target tokens autoregressively.

Contrast with decoder-only: no cross-attention, no separate encoding step. The decoder-only model conditions on its own prefix. The encoder-decoder model conditions on a *separately encoded* source sequence.

---

## Mechanism

Trace one full forward pass through a 1-layer encoder-decoder Transformer:

**Encoder path:**
- Input embeddings + positional encoding → multi-head self-attention (bidirectional) → add & norm → FFN → add & norm → encoder output matrix `H_enc` of shape `(src_len, d_model)`

**Decoder path (training, teacher-forced):**
- Target embeddings + positional encoding → masked multi-head self-attention (causal mask) → add & norm
- Cross-attention: queries from decoder hidden state, keys and values from `H_enc`. No causal mask on cross-attention — the decoder can attend to any encoder position.
- Add & norm → FFN → add & norm → linear projection → softmax over vocab

**Key insight:** Cross-attention keys and values come from a *different* sequence than the queries. This is the only architectural difference between decoder-only and encoder-decoder Transformers. Everything else is shared weights or shared structure.

**Inference difference:** Decoder generates one token at a time. Encoder runs once. Cross-attention reuses the cached `H_enc` at every decoding step. This is why encoder-decoder models can be efficient for long-source, short-target tasks.

---

## Build It

Implement a minimal encoder-decoder Transformer in PyTorch — one encoder block, one decoder block, cross-attention wired end-to-end. Run a forward pass on dummy data (random token IDs for source and target) and print the output logits shape to confirm `(batch, tgt_len, vocab_size)`.

Show the cross-attention layer explicitly: demonstrate that swapping the encoder input changes the decoder output *even when the decoder input is identical*, proving that information flows through the cross-attention bridge.

Exercise hooks:
- **Easy:** Modify the number of attention heads and print the output shape to confirm independence from head count.
- **Medium:** Implement greedy decoding (argmax loop) that generates 5 tokens autoregressively from a fixed encoder output.
- **Hard:** Add a KV-cache for the encoder output and benchmark decoder steps with vs. without cache — print timing.

---

## Use It

Encoder-decoder architectures dominate in any task where the input and output are *different sequences* with different structure: translation, summarization, code generation from specs, SQL from natural language.

**GTM redirect:** In GTM systems, the encoder-decoder pattern maps to any pipeline that encodes unstructured account signals (10-K filings, earnings call transcripts, job postings) and decodes them into structured outputs (ICP fit scores, account briefs, personalized outreach). The encoder bidirectionally contextualizes the raw source material. Cross-attention lets the generation layer selectively pull from those contextualized representations while producing output token by token.

This is the architectural basis for retrieval-conditioned generation systems. When a downstream tool like Clay aggregates firmographic data and a language model conditions on that data to write a personalized email, the information-flow pattern is encoder-decoder — even if the implementation wraps it in a single decoder-only API call with the context stuffed into the prompt. Understanding cross-attention makes clear *why* stuffing works (the prefix tokens act as a pseudo-encoder) and where it breaks (no bidirectional context in the prefix for decoder-only models).

Cluster reference: [CITATION NEEDED — concept: GTM cluster for AI-conditioned content generation in Clay workflows]

---

## Ship It

Exercise hooks:

- **Easy:** Run the provided encoder-decoder forward pass on a real-ish input (tokenize a short English sentence as source, a short French sentence as target using a simple vocab lookup). Print encoder output shape, cross-attention weight shape, and final logit shape.

- **Medium:** Implement a greedy decoder loop that generates tokens one at a time. Stop at a max length or a special end token. Print each generated token at each step to show the autoregressive loop working.

- **Hard:** Build a two-model comparison: encode a source sentence, then decode with *correct* cross-attention vs. *zeroed-out* cross-attention (keys/values set to zeros). Print the KL divergence between the two output distributions at each decoder position. This quantifies exactly how much information flows through the cross-attention bridge.