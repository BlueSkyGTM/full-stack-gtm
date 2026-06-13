# Perplexity and Calibration

## Hook

Your model says "90% confident this lead matches your ICP." It's wrong half the time. Perplexity and calibration are the two diagnostics that tell you whether a model's confidence scores mean anything at all — or whether you're making pipeline decisions on garbage numbers.

## Concept

Perplexity measures how surprised a language model is by the next token, derived from the average negative log-likelihood of a sequence. Calibration measures whether predicted probabilities match observed frequencies — when a calibrated model says "80% likely," it's correct about 80% of the time. These are distinct: a model can have low perplexity (good predictions) but be poorly calibrated (dishonest confidence), or vice versa.

## Demonstration

Compute perplexity from a model's log probabilities over a text sequence. Then build a calibration curve: bin predictions by confidence level, compare predicted probability to actual hit rate. Show what miscalibration looks like versus a calibrated output. Temperature scaling as a post-hoc fix.

## Use It

In GTM, any system that surfaces a confidence score — ICP matching, intent classification, lead scoring, account prioritization — needs calibration checks. If your Clay waterfall or custom scorer labels accounts "high confidence" but conversion doesn't correlate, you have a calibration problem. Map to Zone 2 (ICP & Scoring) or Zone 3 (Signal Detection). [CITATION NEEDED — concept: calibration auditing in GTM scoring workflows]

## Ship It

**Easy:** Compute perplexity on a short text using log probabilities, print the result.
**Medium:** Build a calibration curve from a list of predicted probabilities and binary outcomes, print expected vs. observed accuracy per bin.
**Hard:** Implement temperature scaling on a set of logits, find the optimal temperature that minimizes calibration error, output before/after reliability diagrams.

## Evaluate

- Calculate perplexity given a set of log probabilities and sequence length.
- Read a calibration curve and identify whether a model is overconfident, underconfident, or calibrated.
- Explain why a model with low perplexity can still be poorly calibrated.
- Identify the GTM risk when a lead scoring model is systematically overconfident.
- Describe what temperature scaling does to logits and why it fixes miscalibration.