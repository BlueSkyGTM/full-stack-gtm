# Token and Positional Embeddings

## Hook
Every LLM starts by turning text into numbers. Token embeddings convert discrete tokens into dense vectors. Positional embeddings inject sequence order into architectures that have none. Without both, transformers cannot distinguish "acquired by" from "by acquired" — a distinction that breaks intent signal detection, account classification, and every downstream GTM task that depends on text.

## Concept
Token embeddings map each token ID to a learned dense vector via a lookup matrix. Positional embeddings encode position information so the model knows where each token sits in the sequence. Two families: fixed sinusoidal encodings (deterministic, no learned parameters) and learned positional embeddings (trained with the model). Both are added element-wise to token embeddings before entering the transformer.

## Mechanism
Walk through the embedding matrix shape `(vocab_size, d_model)` and the lookup operation that produces a `(seq_len, d_model)` tensor. Derive the sinusoidal encoding formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` and `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`. Show why addition (not concatenation) preserves both token identity and position. Demonstrate that cosine similarity between positional encodings decays with distance — nearby positions share similar encodings, distant positions diverge.

Exercise hooks:
- **Easy**: Build a token embedding lookup from scratch using a random initialization matrix. Print the resulting vectors for a short input sequence.
- **Medium**: Implement sinusoidal positional encoding from the formula. Plot (text-based) how encoding similarity decays with distance between positions.
- **Hard**: Construct the full input representation (token + positional embeddings) for a real sentence. Show that swapping two tokens produces different final representations even though the token embeddings themselves haven't changed.

## Use It
In GTM contexts, token and positional embeddings determine how your models encode account descriptions, intent signals, and outreach copy. [CITATION NEEDED — concept: specific GTM enrichment pipeline using transformer embeddings for account classification]. When you run firmographic data through an LLM for categorization, the positional embedding is why "competitor to Snowflake" and "Snowflake competitor to" resolve similarly while "acquired Snowflake" and "Snowflake acquired" resolve differently. Foundational for Zone 2 (Signals & Intent) — any signal extraction pipeline that uses transformer models depends on these representations.

## Ship It
Build a minimal embedding pipeline: tokenize a list of company descriptions, generate token embeddings via lookup, apply sinusoidal positional encodings, output the combined tensor. Verify that identical tokens at different positions produce different vectors. Print cosine similarities to confirm positional encoding behavior. All output terminal-visible, no browser dependencies.

## Evaluate
- Compare the dimensionality and information content of one-hot token representations versus learned dense embeddings. State which is more memory-efficient and why.
- Given a sinusoidal positional encoding at position 5, predict whether position 6 or position 50 will have higher cosine similarity. Justify with the encoding formula.
- Diagnose what happens to a transformer's output if positional embeddings are removed entirely: describe what classes of sequences become indistinguishable.
- Explain why positional encodings are added to token embeddings rather than concatenated, referencing the impact on the dimensionality of subsequent attention layers.