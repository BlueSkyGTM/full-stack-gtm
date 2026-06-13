# Tokenized Dataset with Sliding Window

## Explain It
The sliding window converts a flat sequence of token IDs into fixed-length input–target pairs for autoregressive training. Each window of length `context_length` becomes the input; the same window shifted one token to the right becomes the target. This produces overlapping training examples from continuous text, preserving context across boundaries.

## Build It
Implement a PyTorch `Dataset` that accepts a 1D tensor of token IDs, a `context_length`, and a `stride`. The `__getitem__` method returns `(input_chunk, target_chunk)` where `target_chunk = input_chunk` shifted by one position. Print the first three pairs to confirm the offset alignment.

## Show It
Visualize a short token sequence (20–30 tokens) and draw three consecutive sliding windows over it. Highlight the overlap between windows and label which tokens are input vs. target in each pair. Show how `stride < context_length` produces overlapping training samples and how `stride == context_length` produces non-overlapping chunks.

## Use It
This is foundational for Zone 1 (Target). Any fine-tuned model trained on ICP-specific corpus data—support transcripts, sales call transcripts, win/loss notes—requires a tokenized dataset with sliding window as the training data layer. The same mechanism feeds retrieval-augmented generation (RAG) embedding models and custom classification heads that score lead fit or intent signal from raw text.

## Ship It
Wrap the sliding-window dataset in a `DataLoader` with configurable `batch_size`, `num_workers`, and `pin_memory`. Compute the total number of training examples given `len(tokens)`, `context_length`, and `stride`. Write a validation check: assert that no target token falls outside the original token sequence bounds.

## Prove It
- Easy: Given a token sequence of length 100 and `context_length=10`, calculate how many training pairs the dataset produces with `stride=5` vs. `stride=10`.
- Medium: Inspect the output of three consecutive `(input, target)` pairs and verify that `target[:, :-1] == input[:, 1:]` holds for every position.
- Hard: Modify the dataset to support variable-stride sampling—short stride for dense training regions (high-loss zones identified during a previous training run) and long stride elsewhere. Print the effective dataset size before and after.