# HDF5 Tokenized Corpus

## Hook (Beat 1)
Working with tokenized datasets exceeding RAM requires a storage format that supports random access by token position without loading the entire file. HDF5's chunked datasets solve this by storing array slices independently on disk, enabling indexed reads across corpora of billions of tokens.

## Concept (Beat 2)
Covers the HDF5 hierarchical data model (groups, datasets, attributes), chunked storage mechanics for token arrays, compression filters (gzip, LZF), and why a single flat dataset of concatenated token IDs with a secondary index for document boundaries outperforms per-document datasets for training dataloaders.

## Demo (Beat 3)
Build a tokenized corpus writer that concatenates tokenized documents into an HDF5 dataset, stores a document boundary index as a separate dataset, and a reader that retrieves arbitrary spans by token position. Prints token counts, compression ratios, and random-access read latencies to confirm the mechanism.

## Use It (Beat 4)
GTM redirect: building a tokenized corpus of company knowledge-base articles, support tickets, and sales call transcripts is foundational for fine-tuning and retrieval-augmented generation pipelines in **Zone 2 (AI-enhanced outreach and content)**. The document boundary index enables retrieving "which source document did these tokens come from" during inference — a traceability requirement for compliance-aware GTM systems.

## Ship It (Beat 5)
**Easy:** Write a script that tokenizes a directory of `.txt` files and writes them to a single HDF5 file with a document-boundary index. Print total tokens and per-document offsets.
**Medium:** Add gzip compression at chunk size 8192, measure and print the size difference between compressed and uncompressed versions, and benchmark random-access read time for 10 random spans.
**Hard:** Implement a PyTorch `Dataset` subclass that reads from the HDF5 corpus, shuffles at the document-boundary level, and yields fixed-length token sequences with cross-document padding. Print a batch of token IDs and their source document indices.

## Quiz Ready (Beat 6)
Assessment hooks targeting: (1) why chunked storage enables random access without full-file loads, (2) the tradeoff between compression ratio and read latency for different chunk sizes, (3) why a flat concatenated dataset outperforms per-document datasets for training iteration, (4) reading and interpreting document boundary indices to reconstruct source attribution.

---

## Learning Objectives

1. **Build** an HDF5 dataset containing concatenated token IDs and a document boundary index from raw text files.
2. **Configure** chunk size and compression filters and measure their effect on file size and random-access read latency.
3. **Implement** a reader that retrieves arbitrary token spans by position and maps them back to source documents using the boundary index.
4. **Compare** flat-concatenated storage versus per-document HDF5 group structures for training-loop iteration efficiency.
5. **Explain** why HDF5 chunked storage enables random access to token arrays that exceed available RAM.

## GTM Redirect Rules

- Primary redirect: **Zone 2 — AI-enhanced outreach and content** (tokenized corpus construction for fine-tuning and RAG on proprietary GTM data).
- Secondary redirect: Foundational for any pipeline that trains or fine-tunes language models on company-specific text corpora.
- No forced connection: if the practitioner is not building training data, the redirect is "foundational for custom model training" without fabricating a direct GTM workflow.

## What NOT To Do

- Do not introduce h5py before explaining why chunked binary storage solves the out-of-core access problem.
- Do not demonstrate with toy datasets under 1MB — print actual compression ratios and read latencies so the practitioner sees the mechanism's value at scale.
- Do not skip the document boundary index — without it, source attribution is lost and the corpus is unusable for compliance-traceable GTM applications.
- Do not use `h5py.File` in a context where the practitioner cannot observe what happens on disk — always print file size, dataset shapes, and attribute values.

## Code Rules

- All examples use `h5py` and `numpy` only (both available in standard ML environments).
- Every script prints observable output: file size, dataset shape, token spans, compression ratios, or read latencies.
- No comments in code blocks — the mechanism is explained in prose before the code appears.
- Each code block runs standalone in a terminal with `python script.py`.