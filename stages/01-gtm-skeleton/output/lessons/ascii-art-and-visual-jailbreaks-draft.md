# Lesson Outline: ASCII Art and Visual Jailbreaks

## Learning Objectives

1. Construct ASCII art payloads that encode restricted content in visually parseable forms
2. Detect visual jailbreak patterns in user input using character frequency analysis and pattern matching
3. Evaluate a model's vulnerability to ASCII-based adversarial inputs across multiple encoding strategies
4. Implement input sanitization that normalizes or rejects ASCII art before model processing
5. Compare effectiveness of typographic vs. spatial ASCII encoding against a target model

---

## Beat 1: Hook — When Text Becomes an Image

**Description:** Open with a working demo that feeds an ASCII art bomb schematic to a model via the CLI. The model complies because the content filter reads character tokens, not spatial patterns. Establish the core tension: safety classifiers operate on token sequences, but humans (and increasingly models) read spatial arrangements. One sentence on why this gap matters for anyone shipping AI-powered tools that accept freeform input.

**Exercise Hook (Easy):** Run the provided ASCII art prompt through a local safety classifier and observe the confidence score. Then run the same content as flat text. Compare the two scores.

---

## Beat 2: Concept — Spatial vs. Sequential Parsing

**Description:** Explain the mechanism: content moderation pipelines tokenize input as a flat sequence. ASCII art exploits this by distributing restricted semantics across whitespace and character positioning. Cover two encoding families: typographic (using character shape similarity, e.g., "5" for "S", "|" for "l") and spatial (using character grids to form shapes). Show how a model with vision capabilities or strong pattern-matching can decode these arrangements even when the tokenizer does not reassemble them. Name the tool landscape: this is not a single tool problem — it affects any system that passes user input to an LLM without spatial normalization. Reference the adversarial classification from [CITATION NEEDED — concept: taxonomy of visual jailbreak strategies in LLM red-teaming literature].

**Exercise Hook (Medium):** Build a script that takes a restricted word, generates both a typographic substitution and a spatial ASCII encoding, and writes each to a file. Inspect both files to confirm the word is human-readable but token-diffuse.

---

## Beat 3: Detect It — Pattern Matching for Visual Payloads

**Description:** Present a detection algorithm: compute character density per line (high variance = possible spatial art), flag lines with unusual character distributions (e.g., high ratio of punctuation and symbols to alphanumeric characters), and scan for known substitution patterns (leet-speak mappings, Unicode confusable lookalikes). Show working Python that implements a simple density-based detector and a regex-based substitution detector. Output: a risk score per input. This is the mechanism — the tool is whatever input-validation layer sits in front of your model call.

**Exercise Hook (Medium):** Extend the provided detector to flag inputs where line-to-line character density variance exceeds a threshold. Test against the ASCII art payloads from Beat 2.

---

## Beat 4: Use It — Guardrails for AI-Powered GTM Workflows

**Description:** GTM redirect: any AI tool in your GTM stack that accepts user-submitted text — enrichment prompts in Clay, chatbot inputs, form-to-email AI processors — is a surface for visual jailbreaks. Demonstrate how to wire the detector from Beat 3 into a pre-processing function that sits between user input and the LLM call. Show a working Python function that normalizes whitespace, collapses ASCII art regions, or rejects high-risk inputs before they reach the model. This is foundational for Zone 3 (Trust & Compliance) in the GTM cluster map — if you're building AI-powered outreach or intake tools, this is your input sanitization layer.

**Exercise Hook (Medium):** Build a CLI tool that reads a text file, runs the density detector, and either passes the content to an LLM API call or rejects it with a reason. Test with clean text, typographic payloads, and spatial ASCII art.

---

## Beat 5: Ship It — Production Defense-in-Depth

**Description:** Show how to combine three layers: (1) input normalization that strips excessive whitespace and collapses character grids, (2) the density-based detector from Beat 3 as a secondary filter, and (3) output-side monitoring that flags model responses indicating successful jailbreak. Provide working code for a pipeline function that chains all three. Discuss logging: record every rejected input so you can tune thresholds. No marketing language — this is a mitigation, not a solution. Determined attackers will evolve encodings. Note where the mechanism is undocumented: the exact tokenization behavior of proprietary models under spatial input is not publicly specified; we observe and test.

**Exercise Hook (Hard):** Build an evaluation harness that runs a battery of 10 ASCII-encoded payloads against the three-layer pipeline. Measure: how many are caught at each layer, and what slips through. Output a summary table.

---

## Beat 6: Extend It — Adversarial Arms Race

**Description:** Point forward: visual jailbreaks are not limited to ASCII. Unicode confusables, zero-width characters, RTL overrides, and actual image-based jailbreaks (for multimodal models) are the broader category. Suggest the practitioner build a living test suite that evolves with new encoding strategies. Reference: [CITATION NEEDED — concept: benchmark datasets for adversarial visual jailbreaks against multimodal LLMs]. Connect back to GTM: as GTM tools add AI-powered parsing of email bodies, support tickets, and form submissions, the attack surface grows. This is ongoing hygiene, not a one-time fix.

**Exercise Hook (Hard):** Research and implement one additional encoding strategy beyond typographic and spatial ASCII (e.g., Unicode confusables or zero-width character insertion). Add it to your test suite from Beat 5. Measure whether the existing detection layers catch it.

---

**GTM Cluster Mapping:** This lesson is foundational for Zone 3 (Trust & Compliance). Specific application: any Clay waterfall step, AI enrichment node, or customer-facing AI tool that ingests freeform text input requires input sanitization against visual jailbreaks.