# Why Transformers — The Problems with RNNs

## Hook
A sequence model that processes tokens one at a time creates a bottleneck: it can't see the full picture, and it can't be parallelized. This beat opens with a concrete demo showing an RNN-style rolling aggregate vs. a full-context aggregate on a short sequence, and the rolling aggregate misses a critical long-range dependency.

## Concept
Defines the three structural failures of recurrent architectures: vanishing gradients over long sequences, sequential computation that prevents parallel training, and the fixed-size bottleneck vector in encoder-decoder designs. Each failure is named and illustrated with a minimal numerical example.

## Mechanism
Walks through a toy RNN cell implemented from scratch in Python (no framework). Shows gradient magnitude decay over 20 timesteps with printed values at each step. Then shows the same information flowing through a self-attention-style lookup (computed manually with matrix operations) where every position has direct access to every other position. The contrast in gradient magnitude and computational dependency graph is the core mechanism.

## Use It
GTM redirect: foundational for Zone 1 (Data Foundation & Enrichment). The architectural constraints of RNNs explain why earlier generation enrichment tools could only process short text windows, while transformer-based tools can ingest full company profiles, long email threads, and multi-quarter earnings transcripts in a single pass. No forced application — this is architectural literacy that informs tool evaluation later.

## Ship It
Not applicable as a standalone ship exercise. This lesson provides the diagnostic vocabulary for evaluating model selection in later lessons. Exercise hook only: "Given a log of gradient magnitudes from a real RNN run, identify the timestep where gradient signal drops below a useful threshold."

## Evaluate It
Three diagnostic questions: (1) name the three structural problems with RNNs, (2) given printed gradient values from the toy RNN, identify vanishing or exploding behavior, (3) explain in one sentence why self-attention avoids the sequential bottleneck. Exercise hooks: Easy — label each problem from a description. Medium — run the provided RNN code, change sequence length, report where gradients vanish. Hard — modify the toy RNN to use gradient clipping and compare output.

---

**Learning Objectives (draft):**
1. Identify the three structural limitations of recurrent architectures when processing long sequences.
2. Demonstrate gradient decay in a toy RNN by running code and interpreting printed gradient magnitudes.
3. Compare the computational dependency graph of an RNN (sequential) vs. self-attention (direct pairwise access).
4. Explain why the encoder-decoder bottleneck vector loses information over long input sequences.