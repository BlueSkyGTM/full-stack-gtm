# Any-Resolution Vision: Patch-n'-Pack and NaFlex

## Learning Objectives

- Pack patches from variable-resolution images into a single token sequence with a block-diagonal attention mask.
- Implement factorized positional embeddings that generalize across arbitrary image dimensions in the same packed batch.
- Build a first-fit-decreasing bin-packing scheduler that assigns images to batches within a token budget.
- Compare AnyRes tiling, NaFlex packing, and M-RoPE along the axes of token efficiency, positional generalization, and implementation cost.
- Compute token budgets for OCR, chart, and logo enrichment pipelines without downscaling input images.

## The Problem

Transformers expect a sequence. A batch is a stack of equal-length sequences. If your images are 224×224, you get 196 patch tokens every time — padding is zero, job done. Train on 224, infer on 224, never think about resolution again.

The world does not cooperate. A receipt is 480×1920 (1:4 aspect ratio). A chart screenshot is 1080×1920 (9:16). A company logo scraped from a website might be 512×512 or it might be 200×3000. A mobile screenshot is 1170×2532. A scanned document ships at 2480×3508. When you resize all of these to 224×224, three things break. Text becomes unreadable because the horizontal strokes in an 8-pixel-tall character get compressed to sub-pixel width. Content gets cropped because square resize either letterboxes (wasting tokens on padding) or center-crops (throwing away the actual content). And tokens get wasted — a 128×128 thumbnail resized up to 224×224 generates 196 patches of interpolated noise where 64 patches of real signal would have sufficed.

You could process each image at its native resolution individually, one forward pass per image. This works but is computationally wasteful — GPU parallelism exists to process batches, not single images. A batch of one 480×1920 receipt followed by a batch of one 128×128 logo means your GPU sits mostly idle on the second forward pass. The naive solution — pad all images in a batch to the dimensions of the largest — is even worse. If one image in a batch of eight is 2048×2048 and the other seven are 256×256, you generate 16,384 patches per image times eight images = 131,072 tokens, of which 114,688 are padding. That is 87.5% wasted compute.

The actual problem is not "how do we handle variable resolution." It is "how do we pack variable-length sequences into a fixed compute budget without padding waste." This is a bin-packing problem dressed up as a vision problem.

## The Concept

Patch-n'-Pack, introduced in NaViT (Google, 2023) and adopted by SigLIP 2's NaFlex variant (2025), solves this by treating vision as a sequence packing problem. Instead of resizing images to a fixed grid, you extract patches at native resolution and concatenate patches from multiple images into a single transformer sequence. The attention mask prevents cross-image attention. This is the same pattern used in LLM training when multiple short documents are packed into one sequence to maximize GPU utilization — the vision community borrowed it.

Three mechanisms make this work. First, **patch extraction at native resolution**: each image is split into patches of fixed pixel size (typically 16×16), but the *number* of patches varies with image dimensions. A 224×224 image yields 14×14 = 196 patches. A 480×1920 receipt yields 30×120 = 3,600 patches. No resizing, no cropping, no distortion. The patch size is the only fixed constant. Second, **token packing**: patches from multiple images are concatenated into a single sequence, and a block-diagonal attention mask ensures each image's patches can only attend to patches from the same image. Image A's patch 47 cannot see Image B's patch 12. This is enforced by masking, not by sequence separation — the transformer processes one long sequence, but the mask creates isolated attention regions within it. Third, **factorized positional embeddings**: standard ViT learned positional embeddings assume a fixed grid (e.g., 14×14 positions). NaViT uses separate row and column embeddings — position (r, c) gets `row_emb[r] ⊕ col_emb[c]` — which generalizes to any grid dimensions the tables can index. A row index of 5 means the same thing whether the image has 14 rows or 120 rows.

```mermaid
flowchart LR
    A["Image A<br/>224×224<br/>196 patches"] --> D["Concatenate<br/>into sequence"]
    B["Image B<br/>256×320<br/>320 patches"] --> D
    C["Image C<br/>192×192<br/>144 patches"] --> D
    D --> E["Block-diagonal<br/>attention mask<br/>660×660"]
    E --> F["Factorized<br/>positional embeddings<br/>row ⊕ col"]
    F --> G["Single forward pass<br/>660 tokens<br/>0 padding"]
```

NaFlex, SigLIP 2's implementation, extends this with flexible resolution scheduling — the encoder can dynamically choose how many patches to extract per image based on a compute budget, trading detail for throughput at inference time. [CITATION NEEDED — concept: NaFlex's specific dynamic resolution scheduling algorithm and whether it differs from NaViT's approach beyond the factorized embeddings]. What is well-documented: NaViT's packing and masking strategy, factorized positional embeddings, and the training efficiency gains from mixed-resolution batches. What is less documented: NaFlex's architectural modifications beyond NaViT, if any, and how the resolution scheduler decides patch counts at inference time.

The alternative approaches each solve the same problem differently. LLaVA-NeXT's AnyRes tiles high-resolution images into a base image plus sub-images (e.g., a 2×3 grid of 336×336 tiles), processes each tile separately, and concatenates the token outputs — simpler to implement but generates redundant tokens at tile boundaries. Qwen2-VL's M-RoPE replaces absolute positional tables entirely with rotary positional embeddings factorized along temporal, height, and width axes, eliminating the need for learned position tables altogether. Each approach trades implementation complexity for positional generalization.

## Build It

Start with the core operation: extract patches from three images at different resolutions, pack them into one sequence, and build the block-diagonal mask.

```python
import numpy as np

def extract_patches(image, patch_size=16):
    h, w = image.shape[:2]
    n_h = h // patch_size
    n_w = w // patch_size
    cropped = image[:n_h * patch_size, :n_w * patch_size]
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = cropped[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.flatten())
    return np.array(patches), n_h, n_w

img1 = np.random.randn(224, 224, 3)
img2 = np.random.randn(256, 320, 3)
img3 = np.random.randn(192, 192, 3)

packed = []
patch_counts = []
grid_sizes = []
for img in [img1, img2, img3]:
    p, n_h, n_w = extract_patches(img)
    packed.append(p)
    patch_counts.append(len(p))
    grid_sizes.append((n_h, n_w))

sequence = np.concatenate(packed, axis=0)
total_tokens = sequence.shape[0]
mask = np.zeros((total_tokens, total_tokens), dtype=int)

offset = 0
for count in patch_counts:
    mask[offset:offset+count, offset:offset+count] = 1
    offset += count

print(f"Image resolutions: 224x224, 256x320, 192x192")
print(f"Patch grids: {grid_sizes}")
print(f"Patch counts: {patch_counts}")
print(f"Packed sequence shape: {sequence.shape}")
print(f"Attention mask shape: {mask.shape}")
print(f"Mask is block-diagonal: {np.array_equal(mask, np.block([[np.ones((c, c), dtype=int) if i == j else np.zeros((patch_counts[i], patch_counts[j]), dtype=int) for j in range(len(patch_counts))] for i, c in enumerate(patch_counts)]))}")
print(f"Diagonal blocks visible (first 30x30 of mask):")
for row in mask[:30, :30]:
    print(''.join(str(v) for v in row[:30]))
```

Run this and you see the block-diagonal structure: ones in the top-left 196×196 block (image A), ones in the next 320×320 block (image B), ones in the final 144×144 block (image C), zeros everywhere else. The transformer processes 660 tokens in one forward pass with zero padding. Without packing, padding all three images to 256×320 would generate 320 × 3 = 960 tokens — 45% more compute for the same information.

Now add factorized positional embeddings. The key property: row embedding index 3 means "the 4th row of patches" regardless of whether the image has 14 rows or 20 rows. This is what lets a single set of learned tables serve every resolution.

```python
import numpy as np

patch_size = 16
embed_dim = 8
half_dim = embed_dim // 2
max_rows = 256
max_cols = 256

row_table = np.random.randn(max_rows, half_dim) * 0.02
col_table = np.random.randn(max_cols, half_dim) * 0.02

def factorized_pos_emb(n_h, n_w):
    embs = []
    for r in range(n_h):
        for c in range(n_w):
            combined = np.concatenate([row_table[r], col_table[c]])
            embs.append(combined)
    return np.array(embs)

grid_large = (480 // patch_size, 640 // patch_size)
grid_small = (224 // patch_size, 224 // patch_size)

emb_large = factorized_pos_emb(*grid_large)
emb_small = factorized_pos_emb(*grid_small)

packed_emb = np.concatenate([emb_large, emb_small], axis=0)

print(f"480x640 grid: {grid_large} -> {emb_large.shape[0]} patches")
print(f"224x224 grid: {grid_small} -> {emb_small.shape[0]} patches")
print(f"Packed positional embeddings shape: {packed_emb.shape}")
print(f"Max row index used: {max(grid_large[0], grid_small[0])} (table size: {max_rows})")
print(f"Max col index used: {max(grid_large[1], grid_small[1])} (table size: {max_cols})")
print(f"Row 0, Col 0 embedding from large image:  {emb_large[0][:4]}")
print(f"Row 0, Col 0 embedding from small image:  {emb_small[0][:4]}")
print(f"Same embedding for same (r,c) across images: {np.allclose(emb_large[0], emb_small[0])}")
print(f"Row 5, Col 3 in large image equals Row 5, Col 3 in small image:")
idx_large = 5 * grid_large[1] + 3
idx_small = 5 * grid_small[1] + 3
print(f"  large[{idx_large}]: {emb_large[idx_large][:4]}")
print(f"  small[{idx_small}]: {emb_small[idx_small][:4]}")
print(f"  Match: {np.allclose(emb_large[idx_large], emb_small[idx_small])}")
```

The output confirms the property: the same (row, col) pair produces identical embeddings regardless of image size. This is why factorized embeddings generalize to resolutions unseen during training — as long as the image's grid dimensions don't exceed the table size, the lookup works.

Now the piece that makes this practical for production: a packing scheduler. You have a list of images at various resolutions and a max token budget per batch. First-fit-decreasing sorts images by patch count (descending) and greedily assigns each to the first batch with room.

```python
patch_size = 16

resolutions = [
    ("receipt_01", 480, 1920),
    ("chart_landscape", 1080, 1920),
    ("logo_square", 512, 512),
    ("doc_portrait", 2480, 3508),
    ("screenshot_mobile", 1170, 2532),
    ("thumbnail_small", 128, 128),
    ("banner_wide", 200, 3000),
    ("scan_medical", 2048, 2048),
]

images = []
for name, h, w in resolutions:
    n_h = h // patch_size
    n_w = w // patch_size
    count = n_h * n_w
    images.append((name, h, w, n_h, n_w, count))

images.sort(key=lambda x: x[5], reverse=True)

max_budget = 16384
batches = []

for entry in images:
    placed = False
    for batch in batches:
        used = sum(e[5] for e in batch)
        if used + entry[5] <= max_budget:
            batch.append(entry)
            placed = True
            break
    if not placed:
        batches.append([entry])

print(f"Max token budget per batch: {max_budget}")
print(f"Total images: {len(images)}")
print(f"Total batches needed: {len(batches)}")
print()
print(f"{'Batch':>5}  {'Image':<22} {'Resolution':>12} {'Grid':>10} {'Patches':>8} {'Running':>8} {'Util%':>6}")
print("-" * 78)

total_patches = sum(e[5] for e in images)
total_capacity = len(batches) * max_budget

for i, batch in enumerate(batches):
    running = 0
    for entry in batch:
        name, h, w, n_h, n_w, count = entry
        running += count
        util = running / max_budget * 100
        print(f"{i:>5}  {name:<22} {f'{h}x{w}':>12} {f'{n_h}x{n_w}':>10} {count:>8} {running:>8} {util:>5.1f}%")
    batch_total = sum(e[5] for e in batch)
    final_util = batch_total / max_budget * 100
    print(f"{'':>5}  {'--- BATCH TOTAL ---':<22} {'':>12} {'':>10} {batch_total:>8} {'':>8} {final_util:>5.1f}%")
    print()

overall_util = total_patches / total_capacity * 100
print(f"Overall token utilization: {total_patches}/{total_capacity} = {overall_util:.1f}%")
print(f"Wasted tokens to batch gaps: {total_capacity - total_patches}")
```

Run this and you see the packing decisions. The large document (2480×3508 = 33,915 patches) gets its own batch — it exceeds the 16,384 budget so it goes into a batch alone at 100% of a 33,915-token sequence (in practice you'd raise the budget or split the document). The medical scan (2048×2048 = 16,384 patches) fills a batch exactly. Smaller images get combined: the logo, thumbnail, and banner share a batch. The first-fit-decreasing heuristic is not optimal — it is within ~20% of optimal in practice, which is good enough given that optimal bin-packing is NP-hard and your batch assignment needs to happen in milliseconds, not seconds.

## Use It

Patch-n'-pack at native resolution is foundational for Zone 2 enrichment pipelines that process visual signals — company logos, product screenshots, ad creatives, document scans — where downscaling destroys the signal that makes extraction work. Consider a Clay enrichment workflow that ingests a prospect's website screenshot, their LinkedIn company logo, and a PDF of their pricing page. Each arrives at a different resolution. The screenshot might be 1440×900 (landscape), the logo 200×200 (square), the PDF page rendered at 2550×3300 (portrait letter). A fixed-resolution encoder forced to resize all three to 224×224 will read the logo fine (downscaled slightly), will struggle with the screenshot (text becomes blurry), and will make the PDF pricing table completely unreadable (3300 pixels compressed to 224 is a 14.7× reduction — every character becomes sub-pixel).

With patch-n'-pack, the same encoder processes all three at native resolution in a single forward pass. The screenshot generates 5,062 patches, the logo generates 169 patches, the PDF page generates 32,938 patches. The packing scheduler assigns them to batches within the token budget. The logo costs almost nothing in compute — 169 tokens out of a 16,384 budget means it rides alongside other enrichment images with negligible overhead. The PDF page is expensive but the information is preserved: every number in the pricing table is represented by real pixels at real resolution, not interpolated mush.

The GTM application is direct: enrichment accuracy on visual data — OCR quality on scanned business cards, chart data extraction from earnings report screenshots, logo classification for brand monitoring — degrades proportionally to the downscaling factor. A pricing table at 14.7× downscale is not "slightly worse OCR." It is "OCR returns garbage characters because the vertical strokes in digits have been averaged away." Patch-n'-pack eliminates this failure mode entirely. The cost is token budget management: you need the packing scheduler to avoid one 33,000-patch document monopolizing a batch that could have held fifteen logos.

## Ship It

Token budget utilization in a patch-n'-pack pipeline is the visual analog of reply rate drift — it is a pipeline health signal that tells you when your input distribution has shifted. Zone 12 observability means instrumenting the packing scheduler to log not just throughput but utilization metrics: average batch fill percentage, max image resolution per batch, and the ratio of processing time to token count. When average batch utilization drops from 85% to 40% over a week, your enrichment pipeline is receiving images with abnormal aspect ratios — maybe a data source started returning thumbnails instead of full screenshots, or a scraper is hitting a CDN that serves compressed previews. The token budget metric catches this before the downstream model quality metrics do, because the model will silently produce worse embeddings on thumbnails without throwing an error.

In production, the packing scheduler becomes an observability surface. Every batch assignment is a log line: timestamp, image count, total tokens, budget utilization, max resolution, min resolution. You can trace any degraded enrichment result back to the specific batch it was packed into and see whether it shared a batch with an anomalously large image that starved it of attention capacity (attention is masked, so this should not matter — but if your mask implementation has a bug, this is how you find it). The first-fit-decreasing scheduler also produces a natural ordering of images by patch count, which is useful for capacity planning: if your 95th percentile image requires 8,000 patches and your budget is 16,384, you can fit at most two large images per batch. When the 99th percentile jumps to 15,000 patches (someone started uploading uncompressed medical scans to the enrichment pipeline), the scheduler log shows the shift before any dashboard alert fires.

The practical observability pattern: emit a structured log entry per batch with `{"batch_id": ..., "image_count": ..., "total_tokens": ..., "budget": ..., "utilization": ..., "max_patches": ..., "min_patches": ...}`. Compute the rolling mean of utilization over 1,000 batches. If it drops below 60%, your pipeline is wasting compute on underfilled batches — either raise the batch budget or investigate why images are smaller than expected. If the rolling max of `max_patches` exceeds your budget, individual images are being processed alone (or split across batches), which means your throughput drops by the ratio of oversized images. These metrics cost nothing to collect — the scheduler already computes them — and they surface distribution shift hours before model quality degrades enough to notice in downstream task accuracy.

## Exercises

**1. Patch count comparison.** Take five images at resolutions 512×512, 1024×1024, 2048×2048, 480×640, and 128×256. Compute the patch count for each at patch_size=16. Compute the total tokens if all five are padded to 2048×2048 and batched together. Then compute the total tokens with patch-n'-pack (no padding). Print the padding waste ratio.

**2. Mask verification.** Modify the first code block to add a fourth image at 320×480. Rebuild the packed sequence and attention mask. Verify the mask is block-diagonal by checking that `mask[i, j] == 0` for every pair `(i, j)` where token `i` belongs to a different image than token `j`. Print the number of cross-image attention entries that are incorrectly set to 1 (should be 0).

**3. Scheduler comparison.** Implement a worst-fit-decreasing scheduler (assign each image to the batch with the most remaining capacity, not the first batch with room). Run both first-fit-decreasing and worst-fit-decreasing on the eight-image resolution list from Build It. Compare the number of batches and overall utilization. Print which heuristic wins for this specific input.

**4. Positional embedding generalization.** Create a factorized embedding table with `max_rows=64, max_cols=64`. Generate embeddings for a 1024×1024 image (64×64 grid). Then generate embeddings for a 1056×1056 image (66×66 grid). What happens? Print the error. Then increase the table size and confirm the larger image works. This demonstrates the constraint: factorized embeddings generalize to unseen resolutions *up to the table size*.

**5. Naive padding cost.** Write a function that takes a list of image resolutions and simulates naive batch padding (pad all images to the max height and max width in the batch). Compute total tokens for naive padding vs patch-n'-pack. Run it on batches where one image is much larger than the others. Print the waste ratio and identify the threshold at which a single oversized image makes naive padding worse than processing images individually.

## Key Terms

**Patch-n'-Pack** — The strategy of extracting patches at native resolution from multiple images and concatenating them into a single transformer sequence, eliminating resize distortion and padding waste.

**Block-diagonal attention mask** — A mask structure where attention is allowed within contiguous blocks of tokens (one block per image)