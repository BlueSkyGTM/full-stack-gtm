# Anthropic Responsible Scaling Policy v3.0

## Open It

Context-setting hook on why a practitioner building with Claude needs to know Anthropic's internal safety commitments: if the model you're calling hits an ASL threshold, your deployment constraints change. Brief framing of RSP v3.0 as the binding document governing that transition.

## See It

Walk through the actual ASL (AI Safety Level) classification framework with concrete capability thresholds. Show what ASL-2 means today versus what ASL-3 triggers operationally (evaluations, deployment restrictions, safety case requirements). Reference the public RSP v3.0 document directly.

## Teach It

Core mechanisms: the evaluation → classification → containment loop. How Anthropic defines capability thresholds (CBRN, cybersecurity, autonomy), what a "safety case" requires, how Red Lines trigger scaling pauses, and how the Responsible Scaling Officer structure governs the process. Explain what changes for downstream users when ASL levels shift.

## Try It

**Easy:** Map five hypothetical model capability profiles to ASL classifications using the threshold definitions from RSP v3.0. **Medium:** Write a safety case outline for a model that passes cybersecurity eval thresholds but not CBRN thresholds. **Hard:** Draft an internal policy document for your own deployment that mirrors the RSP evaluation → classification → containment loop for a specific GTM AI use case.

## Use It

GTM redirect: foundational for **Zone 1 (Research & Enrichment)** and **Zone 2 (Outreach & Engagement)** — any practitioner running autonomous enrichment agents or multi-step outreach sequences on Claude is building exactly the kind of system the RSP's autonomy evaluations assess. Knowing where the ASL thresholds sit tells you what behaviors will trigger scrutiny or throttling in your tool calls.

## Ship It

Deliverable: a one-page risk classification memo for your current AI-assisted GTM workflow. Maps each automated step to the relevant RSP risk category (cyber, persuasion, autonomy), documents your current eval status, and specifies what operational changes you'd need to make if Anthropic transitions to ASL-3.

---

**GTM cluster mapping:** Foundational — Zone 1 (Research & Enrichment) and Zone 2 (Outreach & Engagement). Direct mechanism connection: autonomous multi-step agents executing research and outreach are the precise use case the RSP autonomy evaluations assess.

**Citation status:** [CITATION NEEDED — concept: Anthropic RSP v3.0 specific ASL threshold numerical values, exact Red Line trigger definitions, Responsible Scaling Officer decision authority scope]