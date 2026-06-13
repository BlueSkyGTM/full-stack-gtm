# Sampling Methods

## Hook (Beat 1)
You have 500K accounts in your CRM. Your enrichment API budget covers 50K lookups this month. Pick the wrong 50K and your ICP model trains on noise. This is the sampling problem — and most teams solve it by guessing.

## Concept (Beat 2)
Core mechanisms: simple random sampling, stratified sampling (by revenue tier, geography, vertical), systematic sampling (every nth record), reservoir sampling (streaming/fixed-memory), and weighted sampling (bias toward high-value signals). Explain each algorithm before naming any library. Cover when each method introduces bias and when that bias is intentional (stratified) vs. accidental (convenience sampling from a sorted CSV).

## Implement (Beat 3)
Working Python functions for each method against a synthetic account dataset. Each function prints input count, output count, and distribution breakdown so the practitioner can observe whether the sample preserved the population's shape. No external dependencies beyond standard library — use `random` module directly so the mechanism is transparent.

## Use It (Beat 4)
GTM redirect: this is the sampling layer inside enrichment waterfalls (Clay, Flex, Relevance) and the selection logic for ICP training sets. Stratified sampling maps to Zone 01 (Targeting) — you segment by firmographic strata before pulling enrichment data. Weighted sampling maps to Zone 02 (Scoring) — you oversample accounts your model is uncertain about. Reservoir sampling applies when you're consuming a real-time intent feed and need a fixed-size buffer.

## Ship It (Beat 5)
Exercise hooks:
- **Easy**: Write a stratified sampler that takes a CSV of accounts and returns equal-sized samples per revenue tier. Print the before/after distribution.
- **Medium**: Implement reservoir sampling over a streaming list of domain strings. Demonstrate fixed memory by processing 100K items with a reservoir of 100.
- **Hard**: Build a weighted sampler that oversamples accounts matching a partial ICP signal (e.g., "has job posting containing 'evaluating vendors'"). Compare model precision between a random 10K training set and a weighted 10K training set on a held-out test set.

## Evaluate (Beat 6)
Quiz hooks (each maps to a testable mechanism):
- Identify which sampling method preserves subpopulation proportions without requiring explicit strata definition.
- Given a sorted CSV (all enterprise accounts first, SMB last), predict what happens with systematic sampling at interval n=100.
- Detect the failure mode: a practitioner samples the first 1K rows of an export for training. Name the bias and the fix.
- Compare reservoir sampling vs. simple random sampling on a stream — state exactly when they produce equivalent distributions and when they don't.
- Given a population that's 80% US, 15% EMEA, 5% APAC, calculate the stratified sample sizes for n=500 with proportional allocation vs. equal allocation, and state which you'd use for ICP model training and why.

---

**GTM cluster context**: Targeting (Zone 01), Scoring (Zone 02), Data Enrichment pipelines. Sampling is the selection mechanism that determines what data enters your GTM systems — garbage in, garbage out starts here.