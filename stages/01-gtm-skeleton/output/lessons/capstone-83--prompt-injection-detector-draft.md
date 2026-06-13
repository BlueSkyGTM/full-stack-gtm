# Capstone 83 — Prompt Injection Detector

## Hook It
A single unchecked user input can reprogram an LLM pipeline. This capstone builds a multi-signal detector that flags injection attempts before they reach your model—critical infrastructure for any system that passes user-supplied text into prompts.

## Ground It
**Learning Objectives**
1. Build a multi-signal prompt injection detector that classifies inputs as safe or potentially malicious.
2. Implement pattern-based detection using regex heuristics for common injection payloads.
3. Implement a classifier-based detection layer using a secondary LLM call.
4. Evaluate detector performance across a test suite of known injection attacks and benign inputs.
5. Configure a scoring pipeline that combines multiple detection signals with tunable thresholds.

**Prerequisites:** Completion of Lessons 76–82, familiarity with regex, API calls to Claude, and basic scoring/normalization logic.

## Explain It
Covers the three dominant detection mechanisms: (1) heuristic pattern matching against known injection signatures (`ignore previous instructions`, `system prompt`, role-resetting phrases), (2) classifier-based detection where a secondary LLM acts as a guard model, and (3) structural analysis—checking for unusual token distributions, instruction-like syntax in user fields, or delimiter smuggling. Explains why no single signal is sufficient and how false positives trade off against safety. Notes where behavior is empirically observed but not formally documented in provider literature.

## Build It
**Exercise hooks:**
- **Easy:** Implement a regex-only detector that flags the top 10 most common injection patterns. Print match results against a provided test set.
- **Medium:** Add a classifier-based secondary check. Build a two-signal scoring pipeline with configurable weights. Run it against a mixed corpus of 20 benign inputs and 20 injection attempts; print precision/recall.
- **Hard (Capstone):** Build a three-signal detector (patterns + classifier + structural analysis). Implement a threshold-tuning loop that sweeps cutoff values and plots false-positive vs. false-negative rates. Output the optimal threshold and final classification report.

## Use It
Any GTM workflow that passes user-submitted data into an LLM pipeline—Clay enrichment formulas that ingest prospect-supplied text, AI email drafters that incorporate reply content, customer-facing support bots—needs input sanitization. This detector plugs in upstream of the prompt assembly step. Relates to Zone 3 (Enrichment) and Zone 5 (Conversion) clusters where external data enters AI-driven flows.

## Ship It
**Exercise hooks:**
- **Easy:** Wrap your detector in a CLI tool that reads stdin and exits with code 0 (safe) or 1 (flagged).
- **Medium:** Package the detector as a Python module with a `check(input_text) -> dict` function returning score, signals, and verdict. Include a config file for threshold tuning.
- **Hard:** Deploy your detector as an HTTP endpoint (using `http.server` or Flask). Send it 50 requests from a test script mixing benign and malicious inputs. Print the confusion matrix and latency stats. Write a one-paragraph threat model documenting what your detector catches and what gaps remain.

---

**GTM Cluster Reference:** Zone 3 Enrichment / Zone 5 Conversion — input sanitization for LLM-driven GTM workflows. [CITATION NEEDED — concept: GTM-specific prompt injection incident case studies in enrichment or conversion pipelines]