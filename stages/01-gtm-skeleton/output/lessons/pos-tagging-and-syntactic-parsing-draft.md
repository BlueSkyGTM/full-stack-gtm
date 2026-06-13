# POS Tagging and Syntactic Parsing

## Hook

A keyword match on "Python" can't tell you whether someone *uses* Python, *teaches* Python, or is *hiring for* Python. Part-of-speech tags and syntactic dependencies encode that distinction. This lesson covers the two layers of grammar analysis that turn flat text into structured relational data you can filter on.

## Learn It

Covers three mechanisms in sequence: (1) tagsets — what POS labels exist and why Universal Dependencies collapsed hundreds of tags into 17; (2) the tagging algorithm — Hidden Markov Models with Viterbi decoding, where transition probability P(NN → VB) and emission probability P("run" | VB) combine to find the most likely tag sequence; (3) dependency parsing — shift-reduce parsers that attach each token to a head token with a labeled arc (nsubj, dobj, amod, etc.), producing a tree rather than a sequence. Distinguishes dependency parsing from constituency parsing; this course uses dependency exclusively because spaCy implements it and downstream GTM extraction tasks need head-modifier pairs, not phrase brackets.

## See It

Working code: load `en_core_web_sm`, run `nlp()` on a sentence, iterate `token.pos_`, `token.dep_`, `token.head.text` and print the results as a readable table. Second example: run the same pipeline on an ambiguous sentence ("I saw her duck") to show how the parser resolves attachment. All output printed to terminal — no visualization libraries.

## Use It

GTM redirect to **enrichment pipeline** (Zone 1). Specific application: given a raw company description pulled from a CRM, extract compound noun chunks for industry tagging, identify past-tense verbs + their direct objects to surface stated capabilities, and flag superlative adjectives as sentiment signals for scoring. Exercise hook: write a function that takes a list of company descriptions and returns a dict per company with keys `industries`, `capabilities`, `sentiment_adjs` — populated only from POS/dependency analysis, no LLM, no keyword lists. Difficulty: medium.

## Ship It

Covers `en_core_web_sm` vs `en_core_web_trf` — speed versus accuracy tradeoff with concrete tokens/second benchmarks. Batch processing via `nlp.pipe()` with `n_process` parameter. When to disable pipeline components you don't need (`disable=["ner", "lemmatizer"]`) to cut latency. Memory ceiling for large document sets: pipe in chunks, don't hold all docs in memory simultaneously. Error boundary: ambiguous POS on short strings (job titles, product names) — the parser was trained on sentences, not fragments; demonstrate the degradation and suggest pre-wrapping fragments in a carrier sentence as mitigation. Exercise hook: benchmark `sm` versus `trf` on 1,000 scraped LinkedIn headlines — print tokens/second and manually spot-check 20 results for tagging accuracy. Difficulty: hard.

## Assess It

Five items: (1) identify which POS tag a token received given a printed spaCy output; (2) given a dependency arc label and two tokens, state which is the head and which is the dependent; (3) predict the output of Viterbi on a two-word sequence given explicit transition and emission probability tables; (4) diagnose why a short phrase ("VP Engineering") produces a flat, uninformative parse and propose the carrier-sentence fix; (5) compare two parser outputs on the same sentence and determine which used the transformer model. All questions grounded in mechanisms from Learn It; no trivia.