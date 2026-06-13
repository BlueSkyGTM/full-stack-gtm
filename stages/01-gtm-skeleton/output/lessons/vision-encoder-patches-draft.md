# Vision Encoder Patches

## Hook
Why a transformer—which expects a sequence of tokens—can ingest an image at all. The answer is patch tokenization: the image becomes a grid of flattened patches, each one treated as a token. This beat opens with a concrete image tensor and the question: "How does this become something a transformer can read?"

## Concept
Define what a patch is geometrically (a small square crop of the input image), how patch size controls the sequence length and compute budget, and what the linear projection layer does to each flattened patch. Cover the relationship between image resolution, patch size, and the resulting token count: `tokens = (H / patch_size) * (W / patch_size)`.

## Mechanism
Walk through the full algorithm: (1) load image as tensor, (2) break into non-overlapping patches via `unfold` or reshape, (3) flatten each patch, (4) linear-project each patch to the embedding dimension, (5) prepend a CLS token, (6) add positional embeddings. Code example takes a synthetic image tensor and prints patch shapes at each stage—observable output, no display required. No comments in code.

## Use It
GTM cluster: **Zone 1 — Signal & Enrichment**. Vision encoders that use this patch mechanism power logo detection, screenshot-based company identification, and document parsing in enrichment pipelines. When a tool like Clay processes a company's homepage screenshot through a vision model to extract firmographic signals, the patches are the input tokens. The redirect: this is the tokenization layer under any vision-based enrichment step.

## Ship It
Exercise hooks:
- **Easy**: Given a synthetic 224×224 tensor with 4 color channels, compute and print the number of patches for patch sizes 16 and 32.
- **Medium**: Implement the full patch extraction and linear projection pipeline on a synthetic image tensor. Print the shape before and after projection. Confirm the token count matches the formula.
- **Hard**: Build a minimal patch tokenizer class that accepts arbitrary image size and patch size, validates that the image is divisible by the patch size, and returns the full sequence of projected patch embeddings plus a CLS token. Print the final tensor shape and the first 5 values of the CLS token.

## Evaluate
Assessment hooks (not full questions):
- Calculate token count given image dimensions and patch size—test the formula directly.
- Predict the shape of the patch embedding matrix before and after linear projection.
- Identify what happens when image dimensions are not divisible by patch size (error vs. padding behavior).
- Compare two patch sizes on the same image: which produces more tokens, and what that means for downstream compute cost in an enrichment pipeline that runs at scale.
- Given a printed tensor shape from the pipeline, trace back to the original image dimensions and patch size.