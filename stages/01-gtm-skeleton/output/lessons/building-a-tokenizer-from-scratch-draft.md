# Building a Tokenizer from Scratch

## Hook

You pass text to an LLM and it responds. But models don't read text — they read integer sequences. That translation layer is the tokenizer, and its behavior determines how your input is chunked, how much context you consume, and why some words cost more tokens than others. Build one from scratch and you stop guessing.

## Concept

**Beat 2: The mechanism.** Start with the problem: text is variable-length, models need fixed vocabularies. Walk through the three tokenizer families — character-level (trivial but long sequences), word-level (rigid, infinite vocabulary), and subword-level (the practical middle). Introduce Byte-Pair Encoding as the algorithm: count pairs, merge the most frequent pair, repeat until vocabulary budget is exhausted. Show how this produces a merge table. Explain encoding as: apply merges in priority order. Explain decoding as: concatenate and detokenize. No tool yet — just the algorithm on paper with a 20-character corpus.

**Exercise hook (easy):** Given a pre-built merge table of 5 merges, manually encode the string `"low lower lowest"` by hand. Show your work at each merge step.

## Demo

**Beat 3: Working code.** Implement a BPE tokenizer in Python from scratch — no imports beyond `collections` and `re`. Build three functions: `train_bpe(corpus, vocab_size)` returns a merge table; `encode(text, merges)` applies merges in order and returns token IDs; `decode(token_ids, vocab)` reconstructs the original string. Train on a small corpus (a paragraph), encode a held-out sentence, decode it back, and print roundtrip equality. Then print the merge table and show how common subwords like `"ing"` and `"th"` get their own tokens.

**Exercise hook (medium):** Modify the training loop to add a frequency threshold — merges only happen if the pair appears more than `N` times. Observe how this changes the merge table and vocabulary for the same corpus. Print side-by-side comparison.

**Exercise hook (hard):** Implement a basic WordPiece tokenizer (the algorithm behind BERT) and compare its merge decisions against your BPE implementation on the same corpus. WordPiece selects merges by likelihood, not frequency — implement that scoring and print the top-10 merges from each algorithm side-by-side.

## Use It

**Beat 4: GTM redirect.** Tokenizers determine how much of your prompt budget each prospect email, company description, or ICP definition consumes. In **Zone 2 enrichment pipelines** — specifically when you're batch-classifying company data through an LLM — a tokenizer lets you estimate cost and truncate inputs before you call the API. Build a cost estimator: tokenize your enrichment inputs, count tokens, multiply by your provider's per-token rate, and flag records that exceed context windows *before* you spend money on them. [CITATION NEEDED — concept: token-cost estimation in GTM enrichment workflows]

**Exercise hook (medium):** Take a CSV of 50 company descriptions (provided). Tokenize each with your BPE tokenizer. Compute estimated cost at GPT-4 pricing ($0.03/1K input tokens). Print total batch cost, median tokens per record, and flag any record exceeding 500 tokens.

## Ship It

**Beat 5: Production wrap.** Package your tokenizer as a CLI tool: `python tokenizer.py train --corpus data.txt --vocab-size 500 --output merges.json` and `python tokenizer.py encode --merges merges.json --text "input here"`. The CLI prints token count, token IDs, and reconstructed text. In a **Zone 3 scoring pipeline**, you'd use this to preprocess incoming lead data before feeding it to a classifier — ensuring consistent tokenization between training and inference.

**Exercise hook (medium):** Build the CLI. Train on a provided corpus of SaaS company descriptions (500+ words). Then encode 10 held-out company descriptions and write the token counts to a JSON file. Confirm roundtrip accuracy is 100%.

## Evaluate

**Beat 6: Check your work.** Five quiz questions grounded in the lesson's code and mechanisms.

**Exercise hook (easy):** Three short-answer questions: (1) Why does BPE merge the most frequent pair first — what happens if you merge a rare pair early? (2) What is the difference between BPE's selection criterion and WordPiece's selection criterion? (3) Why does a tokenizer trained on English text perform poorly on code or Chinese — what specifically fails?

**Exercise hook (medium):** Given a corrupted merge table (two merges swapped in priority order), trace the encoding of a test string and show where the output diverges from the correct encoding. Explain why merge order matters.

---

**Learning Objectives:**

1. Implement byte-pair encoding training from a raw text corpus to produce a merge table.
2. Encode and decode text using a BPE merge table, confirming roundtrip accuracy.
3. Compare BPE and WordPiece merge selection criteria and predict how they differ on the same input.
4. Estimate token-based API costs for a batch of GTM enrichment records.
5. Package a tokenizer as a CLI tool with train and encode subcommands.