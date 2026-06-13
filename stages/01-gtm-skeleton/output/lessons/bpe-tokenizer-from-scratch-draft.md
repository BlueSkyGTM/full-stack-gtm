# BPE Tokenizer From Scratch

## Hook

You're paying per token for every GPT call, but you've never watched a tokenizer actually work. Building BPE from scratch shows you exactly where your tokens come from, why "anticipate" and "anti-" split differently, and where your API budget is leaking.

## Concept

**Beat description:** Walk through the BPE algorithm at the whiteboard level: start with bytes, count adjacent pairs, merge the most frequent pair, repeat until vocabulary budget is exhausted. Cover the difference between byte-level BPE (GPT-2/3/4) and character-level BPE (original). Show how merge order determines token boundaries and why the same word can tokenize differently depending on context.

**Mechanism order:**
1. The frequency-counting loop
2. The greedy merge criterion
3. The deterministic application of learned merges at inference time

## Demo

**Beat description:** Build a minimal BPE tokenizer in ~80 lines of Python that trains on a corpus, learns merges, and encodes/decodes new text. Print the merge table, the step-by-step pair counts at each iteration, and the final token sequence for sample inputs. All output to terminal — observable, no browser.

**Exercise hooks:**
- Easy: Modify the number of merges and print the resulting vocabulary size
- Medium: Add a function that prints the cost difference between two tokenization strategies for the same input
- Hard: Implement BPE dropout (randomly skipping merges during training) and compare vocabulary diversity

## Use It

**Beat description:** Map BPE token counting to GTM cost control. Show how token-aware prompt templates reduce spend in Clay workflows that call OpenAI for lead enrichment. Connect to the GTM cluster: **Zone 02 — AI Operations & Cost Governance**. Demonstrate that rewriting "Reach out to schedule a conversation" versus "Let's meet" produces measurably different token counts and costs at scale.

**GTM redirect:** This is the mechanism behind token-budgeted enrichment pipelines. When a Clay waterfall calls an LLM to score leads, the tokenizer determines your per-record cost. Knowing BPE means you can estimate before you run.

## Ship It

**Beat description:** Build a CLI tool that takes a text file (email template, outbound sequence, etc.), tokenizes it with the from-scratch BPE, reports total tokens, and estimates cost at GPT-4 pricing. Add a flag to suggest shorter rephrasings by identifying sub-optimal token splits.

**Exercise hooks:**
- Easy: Ship the token counter with cost output
- Medium: Add the rephrasing suggestion flag
- Hard: Compare your BPE implementation's token count against `tiktoken`'s GPT-4 encoding on 50 real outreach emails and report the delta

## Pin It

**Beat description:** Consolidate with three recall points: (1) BPE is greedy frequency merge, not semantic chunking — "unhappiness" and "happiness" share no tokens despite sharing a word, (2) merge order is learned from training data and frozen at inference — your tokenizer is a fixed function, (3) token count ≠ character count ≠ word count, and API billing uses token count. Close with a one-sentence gut check: "If I change one word in my prompt template, can I predict whether my token count goes up by 1 or by 3?"

---

**Learning Objectives:**
1. Implement the BPE merge loop that trains a vocabulary from a text corpus
2. Encode and decode text using a learned merge table
3. Calculate API cost differences attributable to tokenization artifacts
4. Compare byte-level versus character-level BPE on the same input
5. Evaluate how vocabulary size affects compression ratio on GTM-relevant text (email templates, lead data)

**GTM Cluster:** Zone 02 — AI Operations & Cost Governance (token-budgeted enrichment, prompt cost estimation in Clay waterfalls)