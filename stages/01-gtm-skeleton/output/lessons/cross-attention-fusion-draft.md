# Cross-Attention Fusion

## Hook

You have two sequences of information — say, a company's firmographic profile and a stream of intent signals — and you need one representation that knows which parts of each matter relative to the other. Self-attention can't do this; it only talks to itself. Cross-attention is the mechanism that lets one sequence interrogate another.

## Learn It

Cross-attention computes attention weights where the queries come from sequence A and the keys/values come from sequence B. Walk through the Q/K/V decomposition, the scaled dot-product, and why this produces a fused representation that preserves which tokens in B are relevant to each token in A — unlike concatenation or averaging, which destroy positional specificity. End with a clear statement of the shape transformations: `(seq_len_a, d_model)` + `(seq_len_b, d_model)` → `(seq_len_a, d_model)`.

## Use It

In GTM, this is the mechanism behind multi-signal enrichment fusion. When a Clay waterfall returns firmographic data from one provider and intent data from another, naive field-merge loses the relational structure between them. Cross-attention lets you weight which intent signals matter for which firmographic attributes — this is the foundation for building a unified account score from heterogeneous enrichment sources. Maps to the **Enrichment Fusion** cluster in Zone 2.

## Build It

Implement single-head cross-attention from scratch in NumPy: two input sequences, learned Q/K/V projection matrices, scaled dot-product attention, output the fused representation. Print shapes at every step and the final attention weight matrix to confirm sequence A is selectively attending to sequence B.

## Ship It

Connect the mechanism to a production pattern: when orchestrating multi-source enrichment (Clearbit firmographics + Bombora intent + web scrape embeddings), cross-attention fusion replaces brittle rule-based merging. Address the practical constraints — you need aligned embedding spaces, a projection layer trained on labeled outcome data (converted / not converted), and a fallback for missing providers. Exercise hook: wire a pre-trained cross-attention model into an enrichment pipeline and log the attention weights for the top-k accounts to inspect which signals drove each score.

## Prove It

Three exercises. **Easy:** Given two pre-embedded sequences and projection matrices, compute cross-attention weights by hand and identify which token in B receives the highest weight for a given token in A. **Medium:** Modify the Build It implementation to support multi-head cross-attention and verify the output shape is unchanged. **Hard:** Train a single-layer cross-attention fusion model on synthetic firmographic + intent embeddings with binary conversion labels, and evaluate whether it outperforms a concatenation-based baseline — report the difference in AUC.