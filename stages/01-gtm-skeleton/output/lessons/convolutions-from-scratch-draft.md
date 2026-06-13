# Convolutions from Scratch

## Beat 1: Hook — The Sliding Window That Changed Everything

The convolution is a sliding dot product. You take a small matrix (kernel), drag it across a larger matrix (input), and at each position multiply element-wise then sum. Every edge detector in Photoshop, every face detector in your phone camera, every OCR pipeline that reads business cards — they all run this exact operation thousands of times. You will implement it from scratch and see exactly what it does to data.

---

## Beat 2: Concept — The Convolution Mechanism

Describe the algorithm before any library. A convolution takes three inputs: a 2D signal (image, heatmap, time-series matrix), a small kernel (3×3, 5×5), and a stride value. The kernel slides across the signal left-to-right, top-to-bottom. At each position, element-wise multiply and sum. That sum becomes one pixel in the output. Define the terms: kernel, stride, padding, output shape formula (`(W - K + 2P) / S + 1`). Explain the difference between convolution and cross-correlation (most ML libraries implement cross-correlation and call it convolution — the kernel is not flipped).

---

## Beat 3: Mechanism — Implement It With No Libraries

Write a pure-Python convolution using nested `for` loops over a small 6×6 input and a 3×3 kernel. Print the input, print the kernel, print the output. Then write the same operation using NumPy array slicing — same kernel, same input, same output. Show that both produce identical results. This makes the operation transparent: the practitioner sees the sliding window with their own eyes, no abstraction hiding the arithmetic.

*Exercise hooks:*
- **Easy:** Run both implementations on the same input; `assert` the outputs match.
- **Medium:** Change the kernel values to three different edge-detection patterns (horizontal, vertical, diagonal); observe what each highlights.
- **Hard:** Add stride and zero-padding as parameters; verify output shape matches the formula.

---

## Beat 4: Use It — Feature Detection in Enrichment Signals

**GTM Cluster:** Zone 1 — Data Enrichment. [CITATION NEEDED — concept: convolution applied to enrichment confidence scores or firmographic feature maps]

The convolution is a pattern detector. In GTM, apply a 1D convolution over a time series of engagement signals (email opens, page views, CRM activity) to detect leading-edge patterns — the "shape" of a heating-up account. A kernel like `[-1, 0, 1]` approximates the first derivative: it flags where engagement is rising. This is the same mechanism signal-processing engineers use on audio; here the "audio" is your pipeline activity stream. The practitioner builds a 1D convolution over a mock engagement timeline and prints which accounts trigger the rising-signal kernel.

*Exercise hooks:*
- **Easy:** Convolve a 1D kernel `[-1, 0, 1]` over a 10-day engagement score series; print the days where the derivative exceeds a threshold.
- **Medium:** Build a 2D convolution over a mock "engagement heatmap" (accounts × days); detect which account-day pairs show rising activity.
- **Hard:** Chain two convolutions — first smooth with `[0.25, 0.5, 0.25]`, then detect edges with `[-1, 0, 1]`. Print the smoothed-then-differenced signal alongside the raw signal.

---

## Beat 5: Ship It — From Scratch to Production Call

**GTM Cluster:** Zone 1 — Data Enrichment.

Build a callable function `detect_engagement_edge(scores, kernel, threshold)` that accepts a list of daily engagement scores, a 1D kernel, and a threshold. It returns the indices where the convolved signal exceeds the threshold. Print the input scores, the convolved output, and the flagged days. This is the same sliding-window pattern Clay uses when it waterfalls enrichment data across providers — each provider result is a "pixel," and the waterfall logic decides whether to stop or continue based on whether confidence crosses a threshold. The practitioner ships a script that could be wired into a CRM automation: "if engagement derivative > threshold, flag account."

*Exercise hooks:*
- **Easy:** Ship the function with the rising-edge kernel; run it on 5 mock accounts; print each account's flagged days.
- **Medium:** Add a `smoothing_kernel` parameter; run smooth-then-detect as a two-step pipeline; compare flagged days with and without smoothing.
- **Hard:** Generalize to 2D: the function accepts a matrix (accounts × days), applies a 2D kernel, and returns a list of `(account_index, day_index)` tuples where the signal exceeds threshold.

---

## Beat 6: Review — What You Built and Why It Matters

Recap the mechanism: convolution is sliding dot-product, kernels are pattern templates, output shape is determined by input size / kernel size / stride / padding. The practitioner implemented it twice (loops and NumPy), applied it to engagement signals, and shipped a callable edge-detection function. The GTM redirect: this is the pattern-detection primitive that underpins signal scoring in enrichment pipelines. Next lesson applies this to multi-channel kernels (stacking multiple kernels to detect multiple patterns simultaneously).