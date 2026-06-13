# Numerical Stability

## Hook
A softmax function silently returns all zeros on a batch of 800+ account scores. The model didn't fail — the numbers just fell off the bottom of float64. This beat opens with that failure and asks: what broke, and why didn't Python throw an error?

## Concept (Raw Material)
IEEE 754 floating-point representation: how real numbers are approximated in finite bits. Covers catastrophic cancellation, overflow, underflow, and the condition number of a function. Explains why `log(a * b)` is not the same as `log(a) + log(b)` in floating point, and why `1e308 * 10` becomes `inf` while `1e-324` becomes `0.0`.

## Mechanism
Three failure modes with detection patterns:
- **Overflow/underflow**: softmax without max-subtraction, log-probability chains. Show the stable vs unstable softmax implementation side by side.
- **Catastrophic cancellation**: subtracting nearly equal numbers. Demonstrate with a variance calculation using naive vs Welford's algorithm.
- **Error accumulation**: summing 1 million small numbers in forward vs sorted order, or via Kahan summation.

Each mechanism includes a runnable Python script that prints the wrong answer, the right answer, and the delta.

## Use It
Lead scoring normalization across a large account universe requires computing softmax or z-scores over thousands of entries. Unstable normalization silently produces rank-order inversions — top accounts get scored lower than mid-tier accounts. This maps to **Zone 2 (Enrichment) and Zone 3 (Scoring)** in the GTM topic map. Exercise: normalize a list of 1000 raw scores using both naive and numerically stable softmax; print the top 10 accounts from each to see the rank inversion. [CITATION NEEDED — concept: GTM scoring normalization and numerical stability failure modes in production enrichment pipelines]

## Ship It
- **Easy**: Rewrite a broken softmax to use max-subtraction. Print before/after outputs on an input array containing values > 700.
- **Medium**: Implement Welford's online algorithm for mean and variance. Compare output against numpy on a synthetic dataset with extreme range (values spanning 1e-10 to 1e10). Print both results and the difference.
- **Hard**: Build a log-space probability accumulator using log-sum-exp. Feed it 10,000 log-probabilities (some very negative). Print the total and verify it sums to 1.0 when converted back to probability space. Compare against naive `exp()` then `sum()`.

## Evaluate
- Explain why softmax output becomes `[nan, nan, nan]` when inputs are `[1000, 1001, 1002]` and does not when inputs are `[0, 1, 2]`.
- Compare the output of forward summation vs Kahan summation on `[1e-16] * 10_000_000`. Print both results.
- Identify which operation causes catastrophic cancellation: `sqrt(a**2 + b**2)` when `a >> b`. Implement and print the result of the stable alternative (`|a| * sqrt(1 + (b/a)^2)`).