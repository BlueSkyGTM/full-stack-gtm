# Capstone 85 — Content Classifier Integration

## Hook It

You've built classifiers in isolation. Now wire one into a real pipeline: inbound content hits an endpoint, gets tagged by category and intent, and routes to the correct GTM action. This is the integration layer between "model works in a notebook" and "model works in a revenue motion."

## Ground It

Prerequisites check: text classification fundamentals, API endpoint construction, webhook handling, and basic data routing. Recap the decision boundary concept — your classifier's confidence threshold determines whether content routes automatically or flags for human review. Reference prior lessons on confidence scoring and threshold calibration.

## Show It

Demonstrate a working content classifier pipeline end-to-end: receive a payload via webhook, run inference against a pre-trained classifier, apply confidence thresholds per category, and emit a routed response. Show the mechanism — feature extraction → scoring → threshold comparison → routing decision. Show the observable output: a log entry with category, confidence, and routing destination. Tool mentions: `scikit-learn` for the classifier, `FastAPI` for the endpoint, `requests` for testing. Mechanism first, tool second.

## Build It

**Easy:** Wire a pre-trained classifier to a local endpoint. Send three test payloads and print categorized output with confidence scores.

**Medium:** Add threshold-based routing logic — high-confidence classifications route to one list, low-confidence flags for review. Output confirms routing destination per payload.

**Hard:** Implement a feedback loop where misrouted content updates a rejection log and recalculates threshold bounds. Print the threshold before and after feedback injection.

## Use It

This is the **Content Routing** mechanism in Zone 02 (Lead Intelligence → Qualification). When inbound content — form submissions, email replies, support tickets — arrives unstructured, a classifier tags it by topic and intent, and the pipeline routes it to the correct GTM motion: sales nurture, SDR outreach, support queue, or discard. The confidence threshold is the control lever: tighten it and more content flags for human review; loosen it and more content routes automatically. [CITATION NEEDED — concept: content routing thresholds mapped to GTM conversion data]

## Ship It

Deploy the classifier endpoint to a persistent environment. Configure monitoring: track classification distribution over time, confidence drift, and routing volume per destination. Set an alert when any category's average confidence drops below its threshold for 50 consecutive classifications — that indicates concept drift or a data quality problem. Production classifiers degrade silently; monitoring catches what accuracy metrics in training do not.