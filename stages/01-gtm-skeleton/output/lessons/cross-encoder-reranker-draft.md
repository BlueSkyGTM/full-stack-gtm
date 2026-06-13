# Cross-Encoder Reranker

## Concept

Introduces the retrieve-then-rerank pattern. A first-stage retriever (BM25, bi-encoder, or keyword match) casts a wide net. The cross-encoder reranker scores each (query, candidate) pair jointly using full self-attention, trading latency for precision at the top of the ranked list.

## Mechanism

Contrasts bi-encoder (embed separately, cosine similarity) vs. cross-encoder (concatenate query + doc, pass through transformer, single relevance logit). Explains why cross-encoders cannot pre-compute embeddings — every pair requires a full forward pass — which is why they sit in stage two, not stage one.

## Implement

Loads a pretrained cross-encoder from `sentence-transformers`, feeds it a query and a list of candidate texts, and sorts by the returned relevance scores. All code runs in terminal with printed ranked output.

**Exercise hooks:**
- Easy: Rerank 10 hardcoded candidate strings against a single query; print sorted results.
- Medium: Pull candidates from a JSON file of company descriptions; rerank against an ICP query; write ranked list to stdout.
- Hard: Implement a minimal cross-encoder from scratch using a Hugging Face classification model (concatenate with `[SEP]`, single-label output), compare its rankings to the library version.

## Use It

Maps to the **account scoring and prioritization** cluster in Zone 1. A practitioner has a list of 200 accounts returned by a filter or search. The cross-encoder reranks those accounts against a natural-language ICP description, surfacing the highest-fit accounts to the top. This is the mechanism behind "AI account scoring" — not a classifier, but a pairwise relevance ranker applied post-retrieval.

## Ship It

Covers latency budgets (cross-encoder over 1,000 candidates at batch size 32), when to cap the candidate set, model distillation options (`cross-encoder/ms-marco-MiniLM-L-6-v2` vs. larger variants), and caching strategies. Flags that cross-encoder scores are not calibrated probabilities — do not threshold them without a held-out validation set.

## Evaluate

Introduces ranking metrics (MRR, NDCG@k) with a small labeled test set. Compares stage-one-only retrieval vs. stage-one-plus-reranker to measure the precision lift. Provides runnable code that computes both metrics and prints the delta.

**Exercise hooks:**
- Easy: Given a query, 10 candidates, and ground-truth relevance labels, compute MRR for reranked vs. unreranked lists.
- Medium: Build an evaluation harness: load a JSON dataset of (query, candidates, relevance_labels), run bi-encoder retrieval then cross-encoder rerank, print NDCG@5 for each stage.
- Hard: Sweep candidate set sizes (20, 50, 100, 200) and plot reranker latency vs. NDCG@5 gain to find the cost-precision knee point.