# Build a Transformer from Scratch — The Capstone

---

## Beat 1: Hook

You've used attention layers, positional encodings, and feed-forward blocks in isolation. Now you wire them together into a complete encoder-decoder architecture — the same architectural skeleton behind BERT, GPT, and every modern LLM. No imports from `torch.nn.Transformer`. Every tensor operation is visible.

---

## Beat 2: Concept

Walk through the full data flow: token embedding → positional encoding → stacked encoder layers (self-attention → add/norm → feed-forward → add/norm) → stacked decoder layers (masked self-attention → cross-attention → feed-forward, each with add/norm) → linear projection → softmax. Draw the residual highways. Explain why decoder self-attention is masked and cross-attention is not. Show where training (teacher forcing) diverges from inference (autoregressive sampling).

---

## Beat 3: Demo

Implement a minimal but complete Transformer in raw PyTorch: `MultiHeadAttention`, `PositionalEncoding`, `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Transformer` — each as a standalone `nn.Module`. Run a forward pass with dummy token indices. Print intermediate tensor shapes at every stage to confirm dimensionality. Generate one greedy decode step to show the autoregressive loop.

---

## Beat 4: Use It

**GTM Redirect — Zone 1: Signal Capture / Intent Classification**

A transformer encoder ingesting a sequence of account activity tokens (page views, email opens, CRM events) can classify buying-intent signals. The self-attention mechanism learns which signal combinations matter — the same mechanism that decides which words in a sentence relate. [CITATION NEEDED — concept: transformer-based intent scoring in GTM platforms]

If the mapping feels forced: this is foundational for Zone 1 signal processing. The architecture is the mechanism; the GTM application is downstream classification.

---

## Beat 5: Ship It

**Easy:** Modify the demo's hyperparameters (d_model, n_heads, n_layers) and run the forward pass. Print and compare output shapes to confirm architectural flexibility.

**Medium:** Replace the dummy vocab with 20 real words. Encode a short sentence pair (source/target). Run one full encoder-decoder forward pass. Print the decoder's logits and extract the top-3 predicted next tokens.

**Hard:** Implement greedy decoding in a loop — feed the decoder's own predicted token back as input for 10 steps. Print the generated token sequence. Measure how attention patterns shift across steps by printing attention weights.

---

## Beat 6: Review

- The Transformer is a stack of identical layers, each containing self-attention and feed-forward sublayers with residual connections and layer normalization.
- The encoder sees the full input sequence; the decoder sees only past tokens (masked self-attention) plus the encoder output (cross-attention).
- Every component you built in previous lessons (attention, positional encoding, layer norm) composes into this single architecture.
- The output is a probability distribution over the vocabulary at each position — training uses parallel teacher forcing; inference uses sequential autoregressive sampling.

---

*GTM cluster: Zone 1 Signal Capture — foundational architecture for sequence classification and intent detection pipelines.*