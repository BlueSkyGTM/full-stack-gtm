## Ship It

**Easy.** Modify the simulator's `block_size` parameter from 64 to 32, then to 128. Observe how density changes. With block_size=32, compression produces twice as many blocks (more compression compute) but selection covers fewer tokens per selected block (less selection compute). Print a comparison table showing density at each block size for a fixed 16K sequence.

```python
import random

random.seed(42)

def nsa_density(seq_len, block_size, top_k, window):
    blocks = seq_len // block_size
    comp = blocks
    sel = top_k * block_size
    win = window
    total = comp + sel + win
    return total / seq_len, total

seq_len = 16_384
window = 512
top_k = 32

print(f"{'Block Size':>12} {'Blocks':>8} {'Comp':>8} {'Sel':>8} {'Win':>8} {'Total/Q':>8} {'Density':>10}")
print("-" * 70)
for bs in [32, 64, 128, 256]:
    d, t = nsa_density(seq_len, bs, top_k, window)
    nb = seq_len // bs
    print(f"{bs:>12} {nb:>8} {nb:>8,} {top_k*bs:>8,} {window:>8,} {t:>8,} {d:>10.4f}")
```

**Medium.** Replace the hardcoded top-k magnitude score with a learnable linear gate. Initialize a weight vector of shape `(num_blocks,)` with random values, compute logits = weights · compressed_block_values, select top-k blocks by logit, and print both the gate weights and the selected indices. Run it three times with different random seeds to see how different gates select different blocks.

```python
import random

random.seed(7)

seq_len = 2048
block_size = 64
top_k = 4

num_blocks = seq_len // block_size

sequence = [random.gauss(0, 1) for _ in range(seq_len)]
compressed = []
for i in range(num_blocks):
    block = sequence[i * block_size : (i + 1) * block_size]
    compressed.append(sum(block) / len(block))

gate_weights = [random.gauss(0, 0.1) for _ in range(num_blocks)]

gate_logits = [gate_weights[i] * compressed[i] for i in range(num_blocks)]
scored = sorted(range(num_blocks), key=lambda i: -gate_logits[i])
selected = sorted(scored[:top_k])

print("Learnable Linear Selection Gate")
print(f"Sequence: {seq_len} tokens, {num_blocks} blocks, top-{top_k}")
print()
print(f"{'Block':>6} {'Compressed':>12} {'Gate Weight':>12} {'Logit':>12} {'Selected':>10}")
print("-" * 56)
for i in range(num_blocks):
    sel = "YES" if i in selected else ""
    print(f"{i:>6} {compressed[i]:>12.4f} {gate_weights[i]:>12.4f} {gate_logits[i]:>12.4f} {sel:>10}")

print(f"\nSelected blocks: {selected}")
print("Selected token ranges:")
for sb in selected:
    print(f"  Block {sb}: tokens [{sb*block_size}:{(sb+1)*block_size}]")
```

**Hard.** Fetch a real SEC 10-K filing in plaintext, chunk it into blocks of 64 sentences (or ~512 tokens each), run all three branches, and print which document sections receive the highest selection scores. Then ask a model to summarize the same filing and compare whether the model's stated key points correspond to the high-scoring blocks.

```python
import urllib.request
import random

random.seed(42)

url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/aapl-20230930.htm"

req = urllib.request.Request(url, headers={"User-Agent": "Lesson Demo lesson@example.com"})
response = urllib.request.urlopen(req, timeout=15)
raw = response.read().decode("utf-8", errors="ignore")

import re
text = re.sub(r"<[^>]+>", " ", raw)
text = re.sub(r"\s+", " ", text).strip()

sentences = re.split(r"(?<=[.!?])\s+", text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

block_size = 64
num_blocks = len(sentences) // block_size
sentences = sentences[: num_blocks * block_size]

print(f"Fetched: Apple 10-K (FY2023)")
print(f"Total sentences: {len(sentences)}")
print(f"Blocks: {num_blocks} (block_size={block_size} sentences)")
print()

block_word_counts = []
for i in range(num_blocks):
    block = sentences[i * block_size : (i + 1) * block_size]
    wc = sum(len(s.split()) for s in block)
    block_word_counts.append(wc)

block_signals = [wc / max(block_word_counts) for wc in block_word_counts]

top_k = min(8, num_blocks)
scored = sorted(range(num_blocks), key=lambda i: -block_signals[i])
selected = sorted(scored[:top_k])

print(f"Top-{top_k} blocks by signal density:")
print(f"{'Block':>6} {'Sent Range':>14} {'Words':>8} {'Signal':>8} {'Preview':>40}")
print("-" * 80)
for idx in selected:
    start_sent = idx * block_size
    end_sent = (idx + 1) * block_size
    preview = sentences[start_sent][:50]
    print(f"{idx:>6} [{start_sent:>4}:{end_sent:>4}] {block_word_counts