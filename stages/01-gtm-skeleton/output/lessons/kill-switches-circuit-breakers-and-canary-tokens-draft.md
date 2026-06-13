# Kill Switches, Circuit Breakers, and Canary Tokens

## Hook

You automated an outreach sequence. It started sending broken links to 4,000 prospects. By the time you noticed, your sender reputation was damaged. This lesson covers the three mechanisms that prevent this class of failure from cascading: kill switches for immediate halt, circuit breakers for threshold-based isolation, and canary tokens for early breach detection.

## Concept

**Kill Switch**: A hard-coded, always-available mechanism to immediately terminate a running process or campaign. Mechanism: a checked flag (environment variable, feature flag, database row) evaluated at each iteration of a loop or before each outbound action. No grace period, no gradual wind-down.

**Circuit Breaker**: A state machine with three states — Closed (normal), Open (blocked), Half-Open (probing). Mechanism: track consecutive failures against a threshold. When threshold is exceeded, state transitions to Open and all requests are rejected without execution. After a cooldown period, one probe request is allowed (Half-Open). Success resets to Closed; failure returns to Open. Prevents cascading failures in dependent systems.

**Canary Token**: A tripwire resource whose access proves unauthorized activity. Mechanism: embed a unique, trackable identifier (URL, DNS name, API key, database record) in a location only accessible through compromise. When the token is touched, an alert fires. Used for intrusion detection, not prevention.

## Demo

Implement all three patterns in a single Python script:

- A `CircuitBreaker` class that wraps a flaky function, tracks failures, transitions through all three states, and prints state transitions to stdout
- A kill switch checked inside a loop that simulates batch processing, terminated by flipping a boolean
- A canary token implemented as an HTTP endpoint that logs access with caller metadata

All code runs in terminal, prints observable output confirming each mechanism fired correctly.

## Use It

**GTM Redirect: Zone 1 — Enrichment Pipeline Reliability (Clay waterfall pattern)**

Circuit breakers apply directly to Clay-style waterfall enrichment. When one data provider returns errors, the circuit breaker prevents wasting API calls on subsequent providers until the upstream recovers. Kill switches apply to automated sequence campaigns — a single checked flag that stops all outbound activity. Canary tokens apply to GTM data security: embed a decoy ICP list or playbook in your CRM; if it's accessed, you know someone is exfiltrating your GTM data.

**Exercise hooks:**
- *Easy*: Add a circuit breaker to a simulated enrichment waterfall. Print which providers were skipped and why.
- *Medium*: Wrap a multi-step outreach sequence with a kill switch. Demonstrate mid-sequence termination.
- *Hard*: Deploy a canary token in a decoy contact record. Write a listener that alerts on access.

## Ship It

Build a production-ready safety harness for any automated GTM process. Requirements: kill switch reads from environment variable (no redeploy needed), circuit breaker wraps external API calls with configurable failure threshold and cooldown, canary token generates unique URLs and logs all hits. Output is a CLI tool that demonstrates all three running simultaneously against a simulated campaign.

**Exercise hooks:**
- *Easy*: Implement kill switch as a CLI flag that halts a running process.
- *Medium*: Add circuit breaker state persistence so restarts don't reset failure counts.
- *Hard*: Generate canary tokens programmatically, embed them in test CRM records, and build an alert endpoint that reports which token was accessed and when.

## Evaluate

**Assessment anchors (quiz questions written against `docs/en.md` objectives):**

1. Compare kill switch vs. circuit breaker: what problem does each solve that the other cannot?
2. Diagram the circuit breaker state machine: what triggers each transition?
3. Given a scenario where an enrichment provider fails 5 times in a row, predict circuit breaker behavior over the next 10 requests with a threshold of 3 and cooldown of 60 seconds.
4. Explain why a canary token that sends email on access is insufficient for real-time breach response.
5. Implement a circuit breaker that differentiates between retryable (500) and non-retryable (401) HTTP errors — only retryable errors increment the failure counter.

---

**Learning Objectives:**

1. Implement a kill switch that terminates a running process based on an external flag
2. Build a circuit breaker with configurable failure threshold, cooldown period, and three-state transitions
3. Deploy a canary token and detect unauthorized access through alert generation
4. Compare the operational tradeoffs between kill switches, circuit breakers, and canary tokens
5. Apply circuit breakers to GTM enrichment pipelines to prevent cascading API failures