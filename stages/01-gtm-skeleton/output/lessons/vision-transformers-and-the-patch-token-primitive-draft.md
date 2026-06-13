# Vision Transformers and the Patch-Token Primitive

## GTM Redirect Rules

- **Cluster:** Zone 3 — Signal Enrichment, sub-cluster: image-based enrichment (logos, screenshots, document classification)
- **Redirect in "Use It":** patch-token embeddings from ViT produce vector representations of company assets (logos, landing pages) that feed into similarity scoring and firmographic inference pipelines
- **Redirect in "Ship It":** foundational for any pipeline that classifies or embeds visual assets at scale; if the visual asset has no GTM signal, this is foundational for Zone 3

---

## Beat 1 — Hook

An image enters a transformer not as pixels but as a sequence of tokens — identical in structure to a sentence. The patch-token primitive is the mechanism that makes this possible. Without it, a transformer cannot ingest spatial data. With it, the same attention mechanism that processes language processes visuals. Every multimodal model — GPT-4V, Gemini, SigLIP — builds on this primitive.

---

## Beat 2 — Concept

**Mechanism: the patch embedding pipeline.**

1. **Partition.** An image of shape `(C, H, W)` is divided into a grid of non-overlapping patches, each of size `P × P`. This produces `N = (H × W) / P²` patches.
2. **Flatten.** Each patch is reshaped from `(C, P, P)` to a vector of length `C × P²`.
3. **Linear project.** A learned projection matrix maps each flattened patch to a token of dimension `D` (the transformer's hidden size). This is mathematically identical to a word embedding lookup — it just operates on continuous vectors instead of discrete indices.
4. **Prepend CLS.** A learned `<CLS>` token is prepended to the sequence. After the transformer, this token's output serves as the image-level representation.
5. **Add position.** Learned (or sinusoidal) position embeddings are added to every token — without these, the transformer has no spatial information, since attention is permutation-equivariant.
6. **Encode.** The resulting sequence of `N + 1` tokens passes through a standard transformer encoder.

**Why this works / doesn't work:**
- No inductive bias for locality or translation invariance (unlike CNNs). The model must learn spatial relationships from data.
- Consequence: ViT requires more pre-training data than a ResNet to match performance, but scales better in the limit.
- The patch size `P` controls the token sequence length. Smaller `P` → more tokens → finer spatial resolution → higher compute cost.

**Key comparison to NLP transformers:** In text, tokenization is discrete (BPE, WordPiece). In vision, tokenization is geometric (fixed grid). The rest of the architecture is identical.

---

## Beat 3 — Demonstration

Implement patch extraction and linear projection from scratch on a synthetic image. Print shapes at every stage to confirm the mechanism.

```python
import torch
import torch.nn as nn

image_channels = 3
image_height = 224
image_width = 224
patch_size = 16
hidden_dim = 768
num_patches = (image_height // patch_size) * (image_width // patch_size)
batch_size = 1

image = torch.randn(batch_size, image_channels, image_height, image_width)

patch_embed = nn.Conv2d(
    in_channels=image_channels,
    out_channels=hidden_dim,
    kernel_size=patch_size,
    stride=patch_size,
)

cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))

tokens = patch_embed(image)
print(f"After Conv2d: {tokens.shape}")

tokens = tokens.flatten(2)
print(f"After flatten: {tokens.shape}")

tokens = tokens.transpose(1, 2)
print(f"After transpose (patch tokens): {tokens.shape}")

cls_tokens = cls_token.expand(batch_size, -1, -1)
tokens = torch.cat([cls_tokens, tokens], dim=1)
print(f"After CLS prepend: {tokens.shape}")

tokens = tokens + pos_embed
print(f"After position embed: {tokens.shape}")

print(f"Sequence length fed to transformer: {tokens.shape[1]}")
print(f"Hidden dim: {tokens.shape[2]}")
```

**Exercise hooks:**
- *Easy:* Change `patch_size` to 32. Predict the new sequence length before running. Confirm with the output.
- *Medium:* Remove the position embeddings. Pass the tokens through a single `nn.TransformerEncoderLayer`. Print the output shape and explain why it is unchanged.
- *Hard:* Implement patch extraction without `Conv2d` — use `torch.Tensor.unfold` or manual reshaping. Verify that the resulting token norms are identical to the Conv2d approach for the same input.

---

## Beat 4 — Use It

**GTM context:** Company logos, landing page screenshots, and product images carry firmographic signal — industry vertical, company stage, design language. ViT embeddings turn these visual assets into vectors that can be clustered, compared, or fed into downstream classifiers.

**The patch-token mechanism is what produces these vectors.** The `<CLS>` token output from a pre-trained ViT is a dense representation of the entire image. Two companies with visually similar landing pages will have vectors close in cosine distance — this is the same distance metric used in text embedding pipelines for ICp matching in Zone 3 enrichment.

**Code: load a pre-trained ViT, embed an image, and compute similarity between two images.**

```python
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import requests
from io import BytesIO

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def embed_image(url):
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    return output[0]

img_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
img_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

vec_1 = embed_image(img_url_1)
vec_2 = embed_image(img_url_2)

similarity = F.cosine_similarity(vec_1.unsqueeze(0), vec_2.unsqueeze(0))
print(f"Cosine similarity: {similarity.item():.4f}")
print(f"Embedding dimension: {vec_1.shape[0]}")
```

**Exercise hooks:**
- *Easy:* Replace the two URLs with two different images. Print the similarity score. Confirm it drops below 1.0.
- *Medium:* Download 5 company logos (or use placeholder images). Compute all pairwise cosine similarities. Print the result as a matrix.
- *Hard:* Cluster the embeddings from 10+ images using k-means (`sklearn.cluster.KMeans`). Print cluster assignments. This is the same pipeline used for logo-based industry classification in Zone 3 enrichment workflows.

---

## Beat 5 — Ship It

**Production consideration:** In a GTM enrichment pipeline, ViT inference runs at scale — hundreds or thousands of company assets per batch. The patch-token mechanism has a concrete cost: sequence length `N = (H/P)²` determines memory and compute inside the transformer's self-attention layers (which scale as `O(N²)`).

**Trade-off in practice:**
- `patch_size=16` (ViT-B standard): 196 tokens per image. Higher fidelity, higher cost.
- `patch_size=32` (efficient variants): 49 tokens per image. Lower fidelity, lower cost.
- For logo classification or landing page similarity, `patch_size=32` often retains enough signal at ~4× lower attention cost.

**Code: batch inference with a pre-trained ViT, demonstrating batched embedding extraction — the pattern used in production enrichment pipelines.**

```python
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import time

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

synthetic_images = [Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)) for _ in range(32)]

batch_tensor = torch.stack([preprocess(img) for img in synthetic_images])
print(f"Batch shape: {batch_tensor.shape}")

start = time.time()
with torch.no_grad():
    outputs = model(batch_tensor)
elapsed = time.time() - start

print(f"Output shape: {outputs[0].shape}")
print(f"Batch size: {batch_tensor.shape[0]}")
print(f"Inference time: {elapsed:.3f}s")
print(f"Throughput: {batch_tensor.shape[0] / elapsed:.1f} images/sec")

norms = torch.norm(outputs[0], dim=1)
print(f"Embedding norm range: {norms.min().item():.2f} – {norms.max().item():.2f}")
```

**Exercise hooks:**
- *Easy:* Change the batch size from 32 to 8 and then to 64. Report throughput for each. Identify the batch size where throughput plateaus or memory errors occur on your hardware.
- *Medium:* Replace `deit_base_patch16_224` with `deit_base_patch32_224`. Compare throughput and embedding dimension. Explain why throughput changes (or doesn't) relative to patch-16.
- *Hard:* Write a function that processes a list of image URLs in batches of `B`, handles download failures gracefully (skip and log), and returns a dictionary mapping URL → embedding vector. This is the core of a Zone 3 visual enrichment service.

---

## Beat 6 — Review

**What the patch-token primitive does:** Converts spatial image data into a sequence of tokens compatible with transformer architectures. The mechanism is: partition → flatten → linear project → prepend CLS → add positions → encode.

**Why it matters for this curriculum:** Every multimodal model that processes images (GPT-4V, SigLIP, Gemini) uses some variant of this patch-token mechanism. Understanding it is foundational for working with vision-language models, image embeddings in enrichment pipelines, and the architecture choices (patch size, sequence length, compute cost) that affect production GTM systems.

**Key relationships to other lessons:**
- Tokenization (NLP) → same role, different input modality
- Position embeddings → required because attention is permutation-equivariant
- Self-attention scaling → `O(N²)` where `N` is number of patches, not pixels — this is why ViT is tractable
- Embedding similarity (Zone 3) → `<CLS>` token output is the vector used for image similarity

**Next lesson in sequence:** Multimodal alignment — how vision tokens and language tokens are combined in models like CLIP and SigLIP.

---

## Learning Objectives

1. **Implement** the patch-token pipeline (partition, flatten, project, prepend CLS, add positions) from scratch using PyTorch primitives and print intermediate shapes to confirm each stage.
2. **Compare** the inductive biases of ViT (no locality prior, data-hungry) versus CNNs (translation invariance built in) and explain when each is preferable.
3. **Compute** cosine similarity between ViT `<CLS>` embeddings of two images and interpret the result as a visual-relevance score applicable to GTM enrichment.
4. **Evaluate** the compute cost of ViT inference as a function of patch size and image resolution, and select appropriate settings for a batched production pipeline.
5. **Explain** why position embeddings are required for vision transformers when self-attention is permutation-equivariant.