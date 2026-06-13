# Anomaly Detection

## Hook
A B2B SaaS company notices their conversion rate doubled overnight. Was it the new campaign, or did a botnet discover their demo form? Anomaly detection is how you distinguish signal from noise—automatically, at scale, without staring at dashboards.

## Concept
An anomaly is a data point that violates the expected distribution of its peers. Three mechanisms detect them: statistical thresholds (how many standard deviations from the mean), isolation (how few splits to separate a point from all others), and density (how isolated a point is relative to its neighbors). Each trades precision for interpretability differently.

## Demonstration
Build a working anomaly detector using Isolation Forest on synthetic account-level data. Show how contamination rate, tree count, and feature selection change which accounts get flagged. Print the flagged rows with their anomaly scores so the practitioner sees the mechanism's output directly.

## Use It
In GTM, anomaly detection applies to form submission quality, inbound lead volume shifts, and pipeline velocity changes. Redirect: this maps to the **Data Enrichment & Quality** cluster in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`. A practitioner can flag accounts with unusual engagement patterns—signals that predict churn or indicate a buyer intent spike—using the same Isolation Forest mechanism demonstrated above.

## Ship It
Exercise hooks:
- **Easy:** Run the provided detector on a CSV of mock form submissions. Identify which rows are flagged and state why (mechanism, not intuition).
- **Medium:** Adjust contamination rate and tree count. Document how the flagged set changes. Write a one-sentence rule for when to increase vs. decrease contamination.
- **Hard:** Feed the detector real output from a Clay enrichment waterfall. Pipe flagged accounts to a Slack alert via webhook. Confirm the alert arrives with the anomaly score in the message body.

## Extend It
Anomaly detection degrades when the baseline distribution shifts (concept drift). A static model trained on Q1 data will produce false positives by Q3. The next step is monitoring drift itself using Population Stability Index or Kolmogorov-Smirnov tests. [CITATION NEEDED — concept: drift detection libraries with CLI-compatible output for production pipelines]