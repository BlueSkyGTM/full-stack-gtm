# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

## Hook

Four organizations produce the majority of publishable alignment research that directly informs how frontier models are evaluated before deployment. If you're building evaluation pipelines or selecting models for production, their outputs are the upstream signal you're implicitly trusting.

## Concept

Covers each organization's mandate, research methodology, and output format: MATS as a talent pipeline producing alignment researchers via structured mentorship; Redwood Research as an independent lab publishing mechanistic interpretability and adversarial training work; Apollo Research focusing on evaluations for deceptive alignment and scheming; METR (formerly ARC Evals) as the de facto standard for dangerous capability evaluations that frontier labs contract before release. Maps the dependencies: METR evaluations gate model releases, Apollo's deceptive alignment research informs what METR tests for, Redwood's interpretability work provides mechanisms Apollo builds threat models from, and MATS trains researchers who end up at all three.

## Demo

Write a script that queries the Anthropic, OpenAI, and Google safety evaluation disclosures (from public model cards / system cards) and cross-references which METR evaluations were run, which Apollo-informed threat models are mentioned, and which Redwood-style interpretability techniques are cited. Output is a structured comparison table printed to terminal.

## Use It

**GTM Redirect:** Foundational for Zone 1 (ICP & Account Intelligence). When evaluating which frontier model to embed in a GTM tool, alignment eval results determine whether a model is safe to deploy in customer-facing workflows. The mechanism: METR evaluation results appear in model cards; those results constrain which use-case categories vendors permit. A practitioner selecting between Claude, GPT-4, and Gemini for automated outbound should be reading the safety eval sections to know what the model was *not* evaluated for — that gap is where your liability sits. [CITATION NEEDED — concept: GTM model-selection safety checklist]

## Ship It

- **Easy:** Pull the latest system card for Claude 3.5 Sonnet and extract every reference to METR, Apollo, or Redwood. Print the count and section headers.
- **Medium:** Build a comparison matrix of METR evaluation categories across three frontier model releases, flagging which evals were skipped or redacted.
- **Hard:** Write an evaluation coverage report that takes a GTM use case (e.g., "autonomous email drafting with access to CRM data") and maps which METR eval categories are relevant, which are missing, and what the gap implies for deployment risk.

## Review

Five assessment questions covering: identifying which organization is responsible for each evaluation type, mapping the research dependency chain (MATS → Redwood → Apollo → METR), reading a model card's safety section for actionable signals, identifying eval coverage gaps relevant to a deployment decision, and distinguishing between capability evaluation (METR) and alignment threat modeling (Apollo).