# Vision Transformer Encoder

## Beat 1: Hook — Why Not Just Use CNNs?

Convolutional networks embed locality by construction. Vision Transformers discard that inductive bias entirely — they treat an image as a sequence of flat patches and learn every spatial relationship from scratch. That sounds like a disadvantage. In practice, at sufficient scale, it isn't. The practitioner needs to know when ViT's tradeoff (more data, less architectural bias) pays off versus when a ResNet is still the correct call.

## Beat 2: Mechanism — From Pixels to Patch Tokens

Three operations, in order: (1) Split the image into fixed-size non-overlapping patches (e.g., 16×16). (2) Linearly project each flattened patch into a D-dimensional embedding. (3) prepend a learnable `[CLS]` token and add sinusoidal or learned positional embeddings so the model can recover spatial order. The resulting sequence is identical in structure to a BERT input — rows of token vectors — and is fed to a standard transformer encoder stack (multi-head self-attention → layer norm → MLP → layer norm, residual connections throughout). The mechanism is: image becomes sequence, sequence attends to itself, `[CLS]` aggregates global information.

## Beat 3: Code — Patch Embedding in Pure NumPy

Build the patch extraction and linear projection from scratch so the practitioner can observe the shape transformations. Input: a single 224×224×3 image (synthetic). Output: a sequence of 196 patch embeddings (14×14 grid, 16×16 patches), printed shapes at each stage. Then implement the same operation using PyTorch's `nn.Conv2d` with kernel size equal to stride — which is how real ViT implementations do it — and confirm both produce identical shapes.

**Exercise hooks:**
- *Easy:* Print the total number of patches for a 224×224 image with patch size 32.
- *Medium:* Implement positional embedding addition and verify that two patches at different positions produce different final vectors.
- *Hard:* Replace the linear projection with a small single-layer MLP and measure the change in embedding variance across patches.

## Beat 4: Code — Full ViT Encoder Forward Pass

Assemble the complete encoder: patch embedding + positional encoding + `[CLS]` token prepending + N transformer blocks (self-attention, layer norm, MLP, residual connections). Run a synthetic batch through and print: (1) intermediate shape after patch embedding, (2) shape after positional encoding, (3) shape after each transformer block, (4) final `[CLS]` output vector. Use PyTorch. No pretrained weights — the point is to trace the data geometry, not get useful representations.

**Exercise hooks:**
- *Easy:* Vary the number of transformer layers and observe how the output shape changes (spoiler: it doesn't — explain why).
- *Medium:* Extract attention weights from the self-attention layers and visualize which patches attend most to the `[CLS]` token.
- *Hard:* Implement a ViT with patch size 8 instead of 16 and benchmark memory usage — explain the quadratic cost in sequence length.

## Beat 5: Use It — Visual Enrichment in GTM Pipelines

GTM redirect: **Enrichment cluster**. ViT encoders (specifically pretrained models like `google/vit-base-patch16-224`) produce vector representations of images that can be used for similarity search, classification, and clustering. In GTM, this maps directly to enrichment workflows: processing company logos from websites, classifying screenshot thumbnails of landing pages, detecting visual patterns in creative assets, or building a similarity index over a prospect's visual brand identity. The mechanism: encode each image to a single `[CLS]` vector, then use cosine similarity or a nearest-neighbor index (FAISS) to cluster, deduplicate, or score visual assets at scale.

The practitioner does not need to train a ViT from scratch for GTM. The correct usage is: load a pretrained encoder, run inference on a corpus of images, store the resulting vectors, and operate on the vectors downstream. This is identical in pattern to text embedding — different modality, same vector-ops pipeline.

**Exercise hooks:**
- *Easy:* Encode 10 synthetic "logo" images using `google/vit-base-patch16-224` and print the cosine similarity matrix.
- *Medium:* Cluster a corpus of 50 website screenshots by visual similarity using ViT embeddings + k-means, report cluster assignments.
- *Hard:* Build a deduplication pipeline that flags near-identical images across a dataset of 100 synthetic creatives using ViT embeddings at a configurable similarity threshold.

## Beat 6: Ship It — From Encoder to Vector Store

Final deliverable: a standalone script that takes a directory of images, encodes each with a pretrained ViT, writes the vectors and metadata to a simple SQLite or JSONL file, and supports a nearest-neighbor query mode (input image → top-k similar images by cosine similarity). The practitioner must demonstrate the pipeline end-to-end on a folder of at least 20 images (synthetic or real) and print: (1) total encoding time, (2) top-3 neighbors for a query image with similarity scores, (3) confirmation that the output file is queryable. This mirrors the exact operational pattern used in GTM enrichment: encode → store → retrieve.

---

## Learning Objectives

1. **Implement** patch embedding from raw image tensors and verify shape transformations at each stage.
2. **Explain** why ViT discards convolutional inductive bias and what it trades in return.
3. **Configure** a full ViT encoder forward pass and trace the data geometry from pixels to `[CLS]` output.
4. **Evaluate** when a ViT encoder is the correct choice versus a CNN for a given image task and dataset size constraint.
5. **Deploy** a pretrained ViT encoder to produce image vectors and execute nearest-neighbor retrieval against a stored corpus.

---

## GTM Redirect Rules

- Primary cluster: **Enrichment** — image vectorization for similarity, clustering, deduplication of visual assets.
- If the practitioner's GTM system does not process images (many don't), the redirect is: **foundational for Zone 02 (Data Enrichment)** — this is a mechanism the practitioner should recognize in vendor capabilities even if they do not implement it directly.
- No forced connection: ViT is not a lead scoring tool. Do not claim it "transforms account intelligence" or similar. It encodes images into vectors. The GTM value is in what you do with those vectors downstream.