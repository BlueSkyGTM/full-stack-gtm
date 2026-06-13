# Classical Metrics

## Hook

You built a classifier. It's 95% accurate. It also missed every positive case. Accuracy lied to you — here's why.

## Concept

The confusion matrix as the source of truth. Deriving precision, recall, F1, and specificity from first principles. Why each metric optimizes for a different cost of error. The precision-recall tradeoff as a dial, not a fact.

## Demo

Build a confusion matrix from raw predictions. Compute precision, recall, F1, and specificity. Show how class imbalance breaks accuracy. Print all metrics side by side to observe which ones move and which ones stay flat when the positive class is rare.

## Use It

Lead scoring produces binary outcomes: qualified or not. A model that flags 100 leads as qualified and gets 5 right has 5% precision — you just burned SDR time on 95 bad conversations. This is the precision-recall tradeoff applied to pipeline. Evaluate your ICP matching or intent signal classifier with these metrics, not accuracy. Relates to signal scoring in [CITATION NEEDED — concept: GTM signal evaluation cluster in gtm-topic-map.md].

## Ship It

Write a metrics report function that takes ground truth and predictions, returns a dictionary with precision, recall, F1, specificity, and the confusion matrix. No libraries — implement from raw counts. This is what you attach to any model evaluation PR.

### Exercises
- **Easy**: Compute precision and recall from a given confusion matrix.
- **Medium**: Build the full metrics report function from raw prediction arrays.
- **Hard**: Given a dataset with 98/2 class split, construct a classifier that achieves >95% accuracy but 0% recall. Explain why this happens.

## Evaluate

Three questions drawn from the mechanics: derive a metric from confusion matrix cells, identify which metric to optimize for a given cost profile (false positives expensive vs. false negatives expensive), and compute F1 from precision and recall values. All testable against the implemented report function.