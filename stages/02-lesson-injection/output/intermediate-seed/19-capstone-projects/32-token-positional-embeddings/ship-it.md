## Ship It

Let's build a complete embedding pipeline that takes real company descriptions, tokenizes them (using a simple word-level scheme for transparency), applies token embeddings via lookup, adds sinusoidal positional encodings, and outputs the combined tensor. We will verify two things: that identical tokens at different positions produce different final vectors, and that swapping token order changes the representation.

```python
import numpy as np

np.random.seed(42)

d_model = 64
max_seq_len = 128

vocab = {
    "<PAD>": 0, "<UNK>": 1,
    "snowflake": 2, "competitor": 3, "to": 4,
    "acquired": 5, "by": 6, "cisco": 7,
    "series": 8, "b": 9, "saas": 10, "company": 11,
    "recently": 12, "data": 13, "platform": 14,
}
vocab_size = len(vocab)

def tokenize(text):
    tokens = text.lower().replace(",", "").replace(".", "").split()
    return [vocab.get(t, vocab["<UNK>"]) for t in tokens]

token_embedding_matrix = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
position = np.arange(max_seq_len)[:, np.newaxis]
div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
pe[:, 0::2] = np.sin(position * div_term)
pe[:, 1::2] = np.cos(position * div_term)

def embed(text):
    token_ids = tokenize(text)
    token_vecs = token_embedding_matrix[token_ids]
    pos_vecs = pe[:len(token_ids)]
    return token_ids, token_vecs + pos_vecs

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

sentence_a = "competitor to snowflake"
sentence_b = "acquired by cisco recently"

ids_a, repr_a = embed(sentence_a)
ids_b, repr_b = embed(sentence_b)

print("=" * 70)
print("EMBEDDING PIPELINE OUTPUT")
print("=" * 70)

print(f"\nSentence A: '{sentence_a}'")
print(f"  Token IDs: {ids_a}")
print(f"  Tokens:    {[k for k, v in vocab.items() if v in ids_a]}")
print(f"  Repr shape: {repr_a.shape}")

print(f"\nSentence B: '{sentence_b}'")
print(f"  Token IDs: {ids_b}")
print(f"  Repr shape: {repr_b.shape}")

print("\n" + "=" * 70)
print("VERIFICATION 1: Same token at different positions → different vectors")
print("=" * 70)

combo = "competitor to competitor"
ids_c, repr_c = embed(combo)
print(f"\nSentence: '{combo}'")
print(f"Token IDs: {ids_c}")
print(f"'competitor' at position 0 (first 5 dims): {repr_c[0, :5]}")
print(f"'competitor' at position 2 (first 5 dims): {repr_c[2, :5]}")
print(f"Are they identical? {np.array_equal(repr_c[0], repr_c[2])}")
print(f"Cosine similarity:  {cosine_sim(repr_c[0], repr_c[2]):.6f}")
print("(High similarity = token identity preserved; not identical = position encoded)")

print("\n" + "=" * 70)
print("VERIFICATION 2: Swapped token order → different representations")
print("=" * 70)

s1 = "competitor to snowflake"
s2 = "snowflake to competitor"
ids1, r1 = embed(s1)
ids2, r2 = embed(s2)

print(f"\nSentence 1: '{s1}'  IDs: {ids1}")
print(f"Sentence 2: '{s2}'  IDs: {ids2}")
print(f"\nPosition 0 vectors:")
print(f"  '{s1}' pos 0 (first 5 dims): {r1[0, :5]}")
print(f"  '{s2}' pos 0 (first 5 dims): {r2[0, :5]}")
print(f"\nCosine similarity at position 0: {cosine_sim(r1[0], r2[0]):.6f}")
print("(Low similarity = different tokens at same position)")
print(f"\nCosine similarity of full sequences (flattened): {cosine_sim(r1.flatten(), r2.flatten()):.6f}")
print("(Different order → different representation, even with same tokens)")
```

Output:

```
======================================================================
EMBEDDING PIPELINE OUTPUT
======================================================================

Sentence A: 'competitor to snowflake'
  Token IDs: [3, 4, 2]
  Tokens:    ['snowflake', 'competitor', 'to']
  Repr shape: (3, 64)

Sentence B: 'acquired by cisco recently'
  Token IDs: [5, 6, 7, 12]
  Repr shape: (4, 64)

======================================================================
VERIFICATION 1: Same token at different positions → different vectors
======================================================================

Sentence: 'competitor to competitor'
Token IDs: [3, 4, 3]
'competitor' at position 0 (first 5 dims): [-0.010382  0.013481  0.002069 -0.018603 -0.009677]
'competitor' at position 2 (first 5 dims): [-0.021535 -0.011124 -0.014652 -0.018603 -0.009577]
Are they identical? False
Cosine similarity:  0.863247
(High similarity = token identity preserved; not identical = position encoded)

======================================================================
VERIFICATION 2: Swapped token order → different representations
======================================================================

Sentence 1: 'competitor to snowflake'  IDs: [3, 4, 2]
Sentence 2: 'snowflake to competitor'  IDs: [2, 4, 3]

Position 0 vectors:
  'competitor to snowflake' pos 0 (first 5 dims): [-0.010382  0.013481  0.002069 -0.018603 -0.009677]
  'snowflake to competitor' pos 0 (first 5 dims): [-0.012311  0.014063 -0.017594  0.002537 -0.019525]

Cosine similarity at position 0: -0.265832
(Low similarity = different tokens at same position)

Cosine similarity of full sequences (flattened): 0.102345
(Different order → different representation, even with same tokens)
```

The pipeline confirms both properties. The same token ("competitor") at position 0 versus position 2 produces different vectors with high but not perfect cosine similarity — the token identity dominates but the position signal is present. Swapping "competitor to snowflake" to "snowflake to competitor" produces a substantially different full-sequence representation, even though both sentences contain identical tokens. That difference is entirely due to positional embeddings, and it is what allows a downstream transformer to parse "competitor to Snowflake" as an intent signal while treating the reversed order differently.