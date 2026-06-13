# OpenAI Preparedness Framework and DeepMind Frontier Safety Framework

## Beat 1: Hook

Two of the three organizations capable of training frontier models published their internal safety governance structures within months of each other. These frameworks encode specific, testable claims about how to measure and contain model risk. You will evaluate those claims on their mechanics, not their marketing.

## Beat 2: Concept

Cover the four mechanisms that both frameworks share despite different naming: (1) capability evaluation thresholds that trigger governance actions, (2) risk categorization taxonomies (CBRN, cybersecurity, persuasion, autonomy), (3) scorecard-based tracking with defined score levels, and (4) escalation protocols that bind engineering teams to commit-to-publish decisions. Then map where the frameworks diverge: OpenAI's Preparedness team operates as an internal eval-and-escalate function with a Safety Advisory Group, while DeepMind's framework ties capability levels to a Responsible Scaling Policy with external governance commitments.

## Beat 3: Demonstration

Build a minimal risk scorecard in Python that mirrors the structure of OpenAI's published scorecard: categories, score levels (low/medium/high/critical), and an aggregation function that produces a composite risk score. Print the scorecard output and demonstrate how changing one category score from medium to critical shifts the composite result. Then implement DeepMind's capability-level classification as a threshold function that takes evaluation results and outputs a required safety level.

## Beat 4: Use It

**GTM Redirect:** Foundational for Zone 4 (AI Governance & Compliance). These frameworks define the vocabulary and risk categories that enterprise buyers reference in procurement due diligence. Any practitioner shipping a model-based product will encounter these categories in customer security reviews. Map framework risk categories to common enterprise AI vendor questionnaire fields.

## Beat 5: Ship It

**Exercise Hooks:**
- *Easy:* Extend the scorecard implementation to read evaluation results from a JSON file and produce a formatted risk report.
- *Medium:* Implement the escalation logic: write a function that takes a scorecard and returns the required governance action (deploy with monitoring → deploy with mitigations → do not deploy) based on the framework's published thresholds.
- *Hard:* Build a simulated "red team eval pipeline" that takes a list of capability test results, classifies each into risk categories using both frameworks, highlights where the two frameworks disagree on required governance level, and outputs the divergence report.

## Beat 6: Digest

Both frameworks reduce to the same mechanism: evaluate capabilities against thresholds, assign risk levels, and bind deployment decisions to those levels. The differences are in governance structure (internal advisory vs. external commitment) and threshold calibration. [CITATION NEEDED — concept: specific threshold values used by each framework for CBRN and cybersecurity risk levels as of latest publication]. Neither framework is enforceable by a third party; both are self-binding commitments. The practitioner takeaway is the scorecard pattern itself, not any specific organization's version of it.

---

**Learning Objectives (for `docs/en.md`):**

1. Compare the governance escalation mechanisms in OpenAI's Preparedness Framework and DeepMind's Frontier Safety Framework.
2. Implement a risk scorecard that maps capability evaluation results to categorized risk levels.
3. Evaluate where the two frameworks produce divergent governance requirements for the same capability profile.
4. Map framework risk categories to enterprise AI procurement due diligence fields.