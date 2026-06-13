# Prompt Injection and the PVE Defense

## The Hook
A crafted email subject line turns your Clay enrichment waterfall into an exfiltration channel. That's not theoretical—it's the trust boundary problem every LLM application inherits. This lesson maps that boundary and builds the defense.

## The Concept
Define prompt injection: direct (user overwrites system instructions) and indirect (third-party data contaminates context). Explain why LLMs cannot natively distinguish "instruction" from "data" — it is all tokens. [CITATION NEEDED — concept: PVE acronym expansion and original source; best current reference is the Simon Willison / Prompt Injection taxonomy work and the NIST AI 100-2e2023 draft]. Establish the mental model: every external input is adversarial until proven otherwise.

## The Mechanism
Walk through the three-layer PVE defense pattern:
1. **Prevention** — structural separation of instruction and data using delimiters, XML tags, or chat-role boundaries.
2. **Validation** — post-generation check: did the output contain data it should not have? Implement a secondary LLM call or rule-based classifier that scores the response for compliance with original intent.
3. **Enforcement** — strip, redact, or reject outputs that fail validation. Rate-limit and log. Treat the LLM call as untrusted.

Show the attack surface reduction at each layer. Provide a working Python script that demonstrates a direct injection attack bypassing a naive prompt, then the same attack failing against PVE-guarded execution. Observable output: attack success boolean before and after defense.

## Use It
GTM cluster: **Zone 1 — ICP Enrichment & Research** (specifically the Clay waterfall that processes company descriptions, email bodies, and LinkedIn scraped data). Every external data field fed into an enrichment prompt is an indirect injection vector. PVE is the pattern you apply before passing scraped data into a Clay "AI enrich" column. Redirect: "this is the PVE defense applied to the Clay waterfall enrichment step — your company description field is untrusted input."

## Ship It
Exercise hooks:
- **Easy:** Run the provided injection attack script. Observe output. Identify which layer of PVE would have caught it.
- **Medium:** Implement Prevention (delimiter-based separation) for a Clay enrichment prompt that processes LinkedIn company descriptions. Submit the guarded prompt and demonstrate it blocks a hidden instruction in the description text.
- **Hard:** Build a full PVE pipeline: a Python function that accepts untrusted text, wraps it with Prevention, runs it through an LLM, validates the output with a secondary classifier call, and enforces rejection on failure. Log attack attempts and false-positive rejections separately.

## The Recap
Prompt injection exploits the instruction-data confusion in transformer architectures. PVE (Prevention, Validation, Enforcement) is a three-layer defense that reduces — but cannot eliminate — the attack surface. The key constraint: no purely prompt-based defense is provably complete; defense-in-depth with logging is the operational standard. Next lesson applies PVE to multi-step agent loops.