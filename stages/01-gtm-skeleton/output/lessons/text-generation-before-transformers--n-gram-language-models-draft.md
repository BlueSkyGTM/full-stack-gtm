# Text Generation Before Transformers — N-gram Language Models

## Hook

You type a sentence into a predictive keyboard. It suggests the next word. That suggestion comes from counting what usually comes next in a corpus. This is the n-gram model — the simplest language model that actually works, and the baseline every transformer is measured against.

## Concept

An n-gram language model estimates P(word_n | word_1, word_2, ..., word_{n-1}) by counting how often that sequence appears in training text, divided by how often the context appears. The Markov assumption truncates the context to n-1 words. Unigrams (n=1) ignore all context. Bigrams (n=2) look at one word of context. Trigrams (n=3) look at two. The model is a lookup table of conditional probabilities derived from raw counts.

## Mechanism

Build a bigram model from a small corpus: tokenize, count bigram frequencies, compute conditional probabilities, generate text by sampling from the distribution. Show the sparsity problem — most possible bigrams never appear in the training data, giving P = 0. Introduce Laplace smoothing (add-1) as the fix. Show perplexity as the evaluation metric: how surprised is the model by held-out text? Lower perplexity = better model.

## Use It

Exercise hooks:
- **Easy**: Build a bigram model from a 100-word corpus and generate 20 words of text. Print the bigram probability table.
- **Medium**: Add Laplace smoothing. Compare generated output with and without smoothing. Print perplexity before and after.
- **Hard**: Build trigram and bigram models on the same corpus. Compare perplexity scores. Identify where the trigram model fails due to sparsity.

GTM redirect: N-gram models are the mechanism behind keyword co-occurrence analysis used in SEO topic clustering and intent classification. This is foundational for **Zone 1 (ICP & Account Intelligence)** — specifically the keyword-density and topic-relevance scoring that platforms like Clearscope or MarketMuse implement for content optimization [CITATION NEEDED — concept: n-gram based topic scoring in SEO tools].

## Ship It

Exercise hooks:
- **Easy**: Wrap an n-gram model in a CLI that takes a seed phrase and generates text. Output to stdout.
- **Medium**: Build a function that computes perplexity on arbitrary held-out text. Test it on two different corpora and print comparison.
- **Hard**: Implement Kneser-Ney smoothing for a bigram model. Compare perplexity against Laplace smoothing on a 10,000-word corpus.

GTM redirect: Perplexity scoring on domain-specific text determines whether a corpus is sufficient for building account-level language models. This connects to **Zone 2 (Signal Capture & Processing)** — scoring whether your scraped account data has enough signal to feed classification pipelines. The mechanism is the same: measure how well your training data predicts held-out examples from the same distribution.

## Evaluate

**Learning Objectives:**
1. Implement a bigram language model that computes conditional probabilities from raw text.
2. Apply Laplace smoothing to handle unseen n-grams and compute updated probabilities.
3. Calculate perplexity on held-out text to compare models.
4. Compare bigram and trigram models on the same corpus, identifying sparsity failure modes.
5. Explain why the Markov assumption limits long-range dependency capture.