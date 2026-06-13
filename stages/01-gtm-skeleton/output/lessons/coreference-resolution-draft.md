# Coreference Resolution

## Beat 1: Concept

Define coreference resolution as the task of identifying all spans in text that point to the same real-world entity. Distinguish between anaphora (backward reference), cataphora (forward reference), and split antecedents. Explain why this matters for any pipeline that extracts structured data from unstructured text: without it, "she" and "the CTO" and "Jane" remain disconnected fragments.

## Beat 2: Mechanism

Walk through the three-stage algorithm: (1) mention detection — span identification of all noun phrases and pronouns, (2) candidate pairing — filter plausible antecedent-mention pairs using syntactic and semantic constraints, (3) clustering — group mentions that corefer using a classification or ranking model. Explain the mention-pair model vs. span-based end-to-end approach (Lee et al., 2017). Cover the standard constraint: an anaphoric pronoun must agree in number and gender with its antecedent, and must be c-command accessible in the syntax tree.

## Beat 3: Code

Build a working coreference pipeline using `spacy` with the `coref` component or the `fastcoref` library. Feed in a multi-sentence paragraph with mixed pronouns and noun phrases. Print each entity cluster with all resolved spans highlighted. Exercise hooks: easy — resolve pronouns in a two-sentence input; medium — detect where coreference fails on ambiguous antecedents; hard — feed a real sales email and extract per-entity mention clusters.

## Beat 4: Use It

GTM redirect: this is the mechanism behind entity normalization in Clay enrichment workflows and signal aggregation in Zone 1. When you ingest intent signals from multiple sources — job postings, press releases, call transcripts — "the company," "they," and "Acme Corp" must resolve to the same account record. Without coreference, your enrichment waterfall produces duplicate or fragmented entity entries. Specific application: resolve person and company mentions across call transcription output before writing to CRM.

## Beat 5: Ship It

Deploy coreference as a preprocessing step in a data enrichment pipeline. Accept raw text from a source (transcript, email body, scraped news), run coreference resolution, and output a normalized entity map with mention counts and confidence scores. Exercise hooks: easy — single-function wrapper around a coref library; medium — batch process a folder of call transcripts and output resolved entity mentions per file; hard — build a two-stage pipeline that runs coreference then feeds resolved spans into a named entity normalization lookup.

## Beat 6: Evaluate It

Metrics: MUC, B³, CEAF, and CoNLL F1 (average of MUC, B³, CEAF₄). Explain why plain accuracy is meaningless here — most mentions do *not* corefer with most other mentions, so a trivial "nothing corefers" baseline scores high precision with zero recall. Show how to compute B³ on a small gold-standard set. Exercise hooks: easy — compute precision/recall on three sentences with given gold clusters; medium — run a coref model on the CoNLL-2012 test split and score with `neuralcoref` or `fastcoref` built-in metrics; hard — compare two coref models on a domain-specific GTM corpus (job postings) and report which handles pronominal resolution better.