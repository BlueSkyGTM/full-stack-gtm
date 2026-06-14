## Ship It

Package the n-gram model as a reusable module: a CLI that takes a seed phrase and generates text, plus a perplexity function that evaluates any held-out text. This is the shape of a real tool — not a notebook experiment, but something you can run on different corpora and compare.

```python
import argparse
import math
import random
import sys
from collections import defaultdict

class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.context_counts = defaultdict(lambda: defaultdict(int))
        self.context_totals = defaultdict(int)
        self.vocab = set()

    def train(self, tokens):
        self.vocab.update(tokens)
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            target = tokens[i + self.n - 1]
            self.context_counts[context][target] += 1
            self.context_totals[context] += 1

    def prob(self, word, context, smoothing="laplace"):
        V = len(self.vocab)
        count = self.context_counts[context][word]
        total = self.context_totals[context]
        if smoothing == "laplace":
            return (count + 1) / (total + V)
        elif smoothing == "none":
            return count / total if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown smoothing: {smoothing}")

    def generate(self, seed, length=15, smoothing="laplace"):
        result = list(seed)
        for _ in range(length):
            context = tuple(result[-(self.n - 1):]) if self.n > 1 else tuple()
            candidates = self.context_counts.get(context, {})
            if not candidates:
                if smoothing == "laplace":
                    candidates = {w: 1 for w in self.vocab}
                else:
                    result.append("[STUCK]")
                    break
            words = list(candidates.keys())
            weights = [self.prob(w, context, smoothing) for w in words]
            next_word = random.choices(words, weights=weights)[0]
            result.append(next_word)
        return result

    def perplexity(self, tokens, smoothing="laplace"):
        log_prob = 0.0
        count = 0
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            target = tokens[i + self.n - 1]
            p = self.prob(target, context, smoothing)
            if p == 0:
                return float('inf')
            log_prob += math.log(p)
            count += 1
        return math.exp(-log_prob / count)

def tokenize(text):
    return text.lower().replace(".", "").replace(",", "").split()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-gram language model")
    parser.add_argument("--seed", type=str, default="the platform",
                        help="Seed phrase for generation")
    parser.add_argument("--n", type=int, default=2, help="N-gram order")
    parser.add_argument("--length", type=int, default=15, help="Words to generate")
    parser.add_argument("--smoothing", choices=["laplace", "none"], default="laplace")
    parser.add_argument("--eval", action="store_true",
                        help="Print perplexity on training data instead of generating")
    args = parser.parse_args()

    import os
    corpus_path = os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "corpus.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            corpus_text = f.read()
    else:
        corpus_text = """
        our platform helps companies scale their revenue operations
        the platform helps sales teams find better leads faster
        revenue operations teams use the platform every day
        companies that scale faster use data to drive revenue
        sales teams need better leads to close deals this quarter
        better data helps sales teams close deals and scale revenue
        """

    train_tokens = tokenize(corpus_text)
    model = NGramModel(n=args.n)
    model.train(train_tokens)

    if args.eval:
        pp = model.perplexity(train_tokens, smoothing=args.smoothing)
        print(f"Model: {args.n}-gram, smoothing: {args.smoothing}")
        print(f"Perplexity on training data: {pp:.2f}")
        print(f"Vocab size: {len(model.vocab)}")
        print(f"Unique contexts: {len(model.context_totals)}")
    else:
        seed_tokens = tokenize(args.seed)
        generated = model.generate(seed_tokens, length=args.length, smoothing=args.smoothing)
        print(" ".join(generated))
```

Run it as:

```bash
python ngram.py --seed "the platform" --n 2 --length 15 --smoothing laplace
python ngram.py --eval --n 2 --smoothing laplace
python ngram.py --eval --n 3 --smoothing laplace
```

The perplexity comparison between 2-gram and 3-gram on the same corpus shows the sparsity tradeoff in production: more context helps only when the training data is large enough to populate that context. This is directly relevant to Zone 2 (Signal Capture & Processing) — when you scrape account data from 10-K filings, LinkedIn descriptions, and press releases, you need to know whether that corpus has enough signal to feed downstream classification pipelines. Perplexity on domain-specific held-out text answers that question. If your scraped account data has high perplexity against your model, the corpus is too sparse — you need more data or a simpler model. The mechanism is identical to evaluating any language model: measure how well your training distribution predicts held-out examples from the same distribution.