# Natural Language Inference — Textual Entailment

## Hook

You have two sentences. You need to know: does one prove the other, contradict it, or say nothing useful? That's the entire problem. And it shows up constantly — routing support tickets, qualifying inbound leads, checking whether a rep's email actually addressed the buyer's stated need. Textual entailment is the formal name for a three-way classification problem: given a premise and a hypothesis, label it **entailment**, **contradiction**, or **neutral**. This lesson builds the mechanism from the labels upward.

---

## Concept

**The three-way decision boundary.** A premise *P* entails a hypothesis *H* if a human reading *P* would infer *H* is true. Contradiction means *H* must be false. Neutral means neither — *H* could be true or false, *P* doesn't constrain it. This is not fuzzy similarity; it's directional logical implication under open-world assumptions.

**Cross-encoder architecture.** Most production NLI models concatenate `[P] [SEP] [H]` and pass the single sequence through a transformer. The classifier head outputs three logits. This architecture — cross-encoder rather than bi-encoder — exists because entailment depends on token-level interactions between premise and hypothesis, not just global semantic similarity.

**The SNLI / MultiNLI lineage.** The training data for most English NLI models comes from image caption pairs (SNLI) or multi-genre sentence pairs (MultiNLI). Knowing the training domain predicts the failure modes — these models struggle with negation, numerical reasoning, and long premises.

---

## Demo

Build a minimal NLI classifier using `transformers` with a pretrained `roberta-large-mnli` checkpoint. Feed three premise-hypothesis pairs — one per label — and print the predicted label and confidence scores. Observable output: the model correctly classifies entailment, contradiction, and neutral for straightforward sentence pairs.

---

## Use It

**GTM cluster: Zone 2 — Enrichment (intent signal extraction).** Given a lead's stated need (premise) and your ICP qualification criteria phrased as hypotheses, run NLI to detect whether the lead's language *entails* fit. Example: lead says "we need to consolidate five spreadsheets into one dashboard"; hypothesis: "this buyer needs a data unification tool." NLI returns *entailment* → strong signal. Neutral → maybe. Contradiction → disqualify or downscore.

This is the mechanism behind intent-classification enrichments in Clay. When you configure a Clay enrichment column that checks whether a lead's recent LinkedIn activity "matches" your ICP, the underlying pipeline is an entailment-style classifier — not keyword search, not cosine similarity, but a directional check: does the source text *imply* the target property?

[CITATION NEEDED — concept: Clay enrichment column using NLI-style entailment for ICP matching]

---

## Drill

**Easy:** Write a function that takes a premise and three candidate hypotheses, runs NLI on each, and returns only the entailed hypothesis. Print results. Observable output confirms the function selects the correct hypothesis.

**Medium:** Build a batch classifier that processes a CSV of support ticket bodies (premise) against a fixed set of resolution categories phrased as hypotheses. Output a new column with the top entailed category and confidence. Print the first five rows.

**Hard:** Implement a simple negation-aware test suite. Construct 10 premise-hypothesis pairs where the hypothesis contains "not" or "never." Measure accuracy against your manual labels. Print a confusion matrix. The point is to observe where the model breaks on negation — and quantify it.

---

## Ship It

**Latency and cost constraints.** Cross-encoder NLI is slow relative to bi-encoder retrieval. For real-time routing (sub-200ms SLA), you need either a distilled model (e.g., `distilbart-mnli-12-1`) or a pre-computed embedding approach that sacrifices directional accuracy for speed. Measure before you ship.

**Calibration.** NLI softmax scores are not calibrated probabilities. A 0.92 entailment score does not mean 92% confidence in the real-world sense. Wrap the model with temperature scaling or Platt scaling if downstream logic threshold-gates on the score.

**Production pattern for GTM.** In a Clay waterfall, NLI sits after data enrichment and before score assignment. The pipeline is: fetch text → construct premise-hypothesis pairs → run NLI → map entailment score to ICP tier. Do not use NLI as a standalone classifier; use it as one signal in a multi-column waterfall.

**GTM redirect.** This is the entailment mechanism behind intent-signal extraction in Zone 2. If you are building ICP qualification, lead routing, or inbound ticket classification in Clay, NLI is the directional check that replaces keyword matching.

---

## Learning Objectives

1. Classify a premise-hypothesis pair as entailment, contradiction, or neutral using a pretrained NLI model.
2. Explain why cross-encoder architecture is used for NLI instead of bi-encoder similarity.
3. Implement a batch entailment classifier that maps free-text inputs to predefined categories.
4. Evaluate NLI model failure modes on negation-bearing inputs using a confusion matrix.
5. Configure an entailment-based signal as an enrichment step in a Clay waterfall for ICP qualification.