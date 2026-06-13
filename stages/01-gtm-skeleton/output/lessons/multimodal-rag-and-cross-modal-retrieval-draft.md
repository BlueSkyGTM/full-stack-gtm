# Multimodal RAG and Cross-Modal Retrieval

---

## Beat 1 — Hook It

A revenue team's knowledge base is never text-only. Competitive battlecards live in slide decks with comparison screenshots. Product docs embed architecture diagrams. A rep asks a natural-language question and the answer is inside an image no text-based RAG pipeline can see. Single-modality retrieval silently fails on the highest-value content.

## Beat 2 — Map It

Survey the architecture: shared embedding spaces (contrastive image-text pretraining), the dual-encoder pattern that CLIP operationalizes, multimodal chunking strategies (document → image regions + text blocks + table extractions), and fusion points (early vs. late). Distinguish same-modality retrieval from cross-modal retrieval — text query → image result is a different mechanism than text query → text result, even when both live in one index.

## Beat 3 — Build It

Construct a minimal cross-modal retrieval system. Encode a set of images and a set of text passages into a shared embedding space using a contrastive vision-language model. Index all vectors in a single store. Accept a text query, retrieve the top-k nearest neighbors regardless of modality, and return results with modality labels.

**Exercise hook (medium):** Build the dual-encoder pipeline. Encode 5 text passages and 5 image file paths (or URLs) into a shared vector space. Run a text query and confirm the system returns both text and image results ranked by cosine similarity.

**Exercise hook (hard):** Replace the flat index with a modality-aware index that stores a `modality` field and a `source` field per vector. Write a retrieval function that accepts a `filter_modality` parameter so a caller can restrict results to "text only," "image only," or "any."

## Beat 4 — Run It

Execute cross-modal queries against the built index and observe three failure modes: modality bias (text results always outrank image results due to embedding scale mismatch), missing chunk boundaries (a table split across two image chunks), and hallucinated relevance (high cosine similarity but semantically unrelated content). Measure recall@k across modality boundaries.

**Exercise hook (easy):** Run 5 text queries against the mixed-modality index. Print the modality of each result alongside its similarity score. Identify which query produces the most cross-modal results and which stays within the query modality.

## Beat 5 — Use It

**GTM Redirect — Zone 2: Signal Intelligence**

A competitive intelligence workflow ingests competitor pitch decks (PDF), product screenshots (PNG), pricing pages (HTML), and win/loss call transcripts (text). Cross-modal RAG lets a rep ask "how does Vendor X position themselves against us on security?" and retrieve the relevant comparison slide — an image — alongside the transcript excerpt where a prospect reacted to it. This is the retrieval pattern underneath multimodal signal pipelines. Clay's enrichment waterfall can feed structured text into this pipeline; the cross-modal retrieval layer is what makes image-based signals queryable alongside text-based signals.

**Exercise hook (medium):** Simulate a competitive intelligence corpus: 3 text snippets (call transcripts), 2 image descriptions (screenshots encoded as image embeddings), and 1 table (encoded as text). Run the query "competitor pricing model weaknesses." Print each result's source type, similarity score, and whether it came from a cross-modal retrieval. Evaluate whether image results contain information the text-only results missed.

## Beat 6 — Ship It

Production concerns: embedding dimension alignment between vision and text encoders, index refresh cadence when images update, latency budget for dual-encoder inference at query time, and evaluation methodology when ground truth spans modalities. Define a cross-modal recall metric — given a text query where the gold answer is in an image, does the retriever surface it within top-k?

**Exercise hook (hard):** Implement a cross-modal evaluation harness. Create a labeled test set of 10 query–result pairs where 5 gold results are text and 5 are images. Run retrieval. Compute recall@3, recall@5, and a modality-stratified recall (recall on text-gold separately from recall on image-gold). Print a report showing where the system breaks along modality lines.

---

**GTM Cluster:** Zone 2 — Signal Intelligence  
**Prerequisites:** Vector retrieval fundamentals, embedding spaces, basic RAG pipeline  
**Tools introduced:** CLIP dual-encoder architecture (mechanism: contrastive image-text pretraining produces a shared latent space), `sentence-transformers` with `clip-ViT-B-32` (implements the dual-encoder for local inference), Chroma or FAISS (stores mixed-modality vectors with metadata filters)