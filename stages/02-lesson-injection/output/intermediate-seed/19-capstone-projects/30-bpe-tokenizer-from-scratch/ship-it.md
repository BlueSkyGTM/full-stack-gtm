## Ship It

Build a CLI tool that reads a text file, tokenizes it with your BPE tokenizer, reports total tokens, and estimates cost. Save this as `token_cost.py`:

```python
import sys
from collections import Counter

class BPETokenizer:
    def __init__(self, num_merges=80):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _get_pair_counts(self, ids):
        counts = Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i + 1])] += 1
        return counts

    def _apply_merge(self, ids, pair, new_id):
        merged = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def train(self, corpus):
        ids = list(corpus.encode("utf-8"))
        for step