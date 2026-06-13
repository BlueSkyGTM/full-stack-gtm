# Sequence-to-Sequence Models

## Learning Objectives

1. Implement a character-level encoder-decoder that maps variable-length input to variable-length output.
2. Compare teacher forcing against autoregressive decoding and predict when each causes training instability.
3. Diagnose the information bottleneck in a fixed-length context vector.
4. Measure seq2seq output with exact-match accuracy and character-error rate.
5. Evaluate whether a seq2seq architecture is appropriate for a given transformation task versus alternatives.

---

## Beat 1 — Hook It

The problem: standard models assume fixed input size and fixed output size. Real tasks—translation, summarization, reformatting—demand variable-length in and variable-length out. Seq2Seq is the architecture pattern that solves this by splitting the work into two phases: compress, then generate.

---

## Beat 2 — Ground It

Mechanism breakdown: the encoder reads the full input sequence and collapses it into a single fixed-size context vector (final hidden state). The decoder takes that vector and generates output tokens one at a time, feeding each prediction back as the next input. Training uses teacher forcing (ground-truth previous token); inference uses autoregressive feedback (model's own prediction). The bottleneck: everything the decoder needs must fit in that one context vector. This is the limitation that motivates attention mechanisms in the next lesson.

---

## Beat 3 — Show It

Working code: character-level seq2seq in PyTorch that learns to reverse input strings (e.g., "hello" → "olleh"). Includes encoder GRU, decoder GRU, training loop with teacher forcing, and greedy inference loop. Prints training loss over epochs and sample predictions to confirm the model learned the transformation.

---

## Beat 4 — Build It

- **Easy:** Run the provided reversal model on a new set of test strings. Print input, expected output, and actual output. Report character-level accuracy.
- **Medium:** Modify the task from "reverse" to "sort characters alphabetically" (e.g., "dbca" → "abcd"). Retrain and compare convergence speed against the reversal task.
- **Hard:** Implement scheduled sampling—start with 100% teacher forcing and linearly decay to 0% over training. Compare final accuracy and exposure bias against fixed teacher forcing.

---

## Beat 5 — Use It

Seq2Seq is the mechanism behind text-style transfer: the same content rewritten for a different audience, tone, or format. In GTM, this is the pattern for transforming a technical value proposition into a plain-language outreach message, or converting a long-form case study into a 3-line email hook. The encoder captures "what was said"; the decoder generates "how to say it differently." This maps directly to personalized messaging workflows in [CITATION NEEDED — concept: GTM content personalization cluster, message rewriting at scale].

---

## Beat 6 — Ship It

Production seq2seq deployments must handle: input length variance (pad and mask correctly), inference latency (autoregressive decoding is inherently sequential—O(output_length) forward passes), and error propagation (one bad prediction feeds into the next). Greedy decoding is the baseline; beam search improves quality at cost of latency. For GTM message generation at scale, batch inference and output length capping are necessary to control cost and throughput. Note: modern production systems use transformer-based encoder-decoders (T5, BART) rather than RNN-based seq2seq—the mechanism is the same, the substrate is different.