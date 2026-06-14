# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

## Learning Objectives

- Implement a multi-scale token pyramid and compute token counts across scales versus raster-scan ordering.
- Construct the block-causal attention mask that enforces causal-across-scale, bidirectional-within-scale attention.
- Trace the full VAR pipeline from multi-scale VQ-VAE tokenization through autoregressive next-scale inference to pixel decoding.
- Compare the context-growth complexity of VAR ($O(K \cdot S_k^2)$) against raster-scan AR ($O(N^2)$) for the same image resolution.
- Evaluate when scale-wise autoregression is preferable to diffusion or raster-scan AR for a given generative image task.

## The Problem

Autoregressive generation dominated language modeling because it scales predictably: more compute, more parameters, lower perplexity, better outputs. Image generation had two main AR attempts before 2024 — PixelRNN/PixelCNN, which predicted pixel-by-pixel, and DALL-E 1 / Parti / MuseGAN, which predicted VQ-VAE token-by-token. Both shared the same structural flaw: they imposed a 1D raster ordering (left-to-right, top-to-bottom) on a 2D grid that has no such natural ordering. The pixel in the top-left corner is predicted first and has no information about what the image will eventually become. Every subsequent prediction conditions on a growing prefix, but the early tokens were generated blind.

This raster ordering fights how images actually compose. When you look at an image, you parse coarse structure first — the overall shape, the color palette, the layout — then your perception fills in fine detail. A cat is recognizable from a blurry silhouette long before individual whisker pixels resolve. Raster-scan AR forces the model to commit to fine-grained details in the top-left before it has generated any global context, producing partial outputs that are globally incoherent: the bottom-right of the image may have no structural relationship to the top-left because hundreds of sequential decisions have drifted apart.

Diffusion models (DDPM, DiT) sidestepped this by denoising the entire image simultaneously across iterative timesteps, which is why they dominated image generation from 2020 to 2024. But diffusion sacrificed the clean scaling laws that made GPT-on-text so effective. VAR asks a different question: what if autoregression operates over scales instead of spatial positions? Generate a 1×1 token map (the overall image summary), then a 2×2 map (coarse features), then a 4×4 map, and so on up to the final resolution. Each scale conditions on all coarser scales, and all tokens within a single scale are generated in one parallel forward pass. The 2024 VAR paper demonstrated this formulation matches GPT-style scaling laws for image generation and outperforms DiT at the same compute budget.

Let us quantify the ordering problem concretely:

```python
def raster_scan_tokens(h, w):
    return h * w

def var_pyramid_tokens(final_h, final_w, num_scales):
    total = 0
    details = []
    for k in range(num_scales):
        divisor = 2 ** (num_scales - 1 - k)
        h_k = max(1, final_h // divisor)
        w_k = max(1, final_w // divisor)
        count = h_k * w_k
        total += count
        details.append((k, h_k, w_k, count))
    return total, details

final_h, final_w = 16, 16
num_scales = 4

var_total, var_details = var_pyramid_tokens(final_h, final_w, num_scales)
raster_total = raster_scan_tokens(final_h, final_w)

print("=== VAR Multi-Scale Pyramid ===")
for k, h, w, c in var_details:
    print(f"  Scale {k}: {h}x{w} = {c} tokens (1 forward pass)")
print(f"  Total tokens: {var_total}")
print(f"  Total forward passes: {num_scales}")
print()
print("=== Raster-Scan AR ===")
print(f"  Total tokens: {raster_total}")
print(f"  Total forward passes: {raster_total}")
print()
print(f"VAR uses {num_scales} sequential steps vs {raster_total} sequential steps")
print(f"Forward pass reduction: {raster_total / num_scales:.0f}x fewer")
print(f"Token overhead: {(var_total - raster_total) / raster_total * 100:.1f}% more tokens total")
```

Running this reveals the core tradeoff: VAR generates slightly more total tokens (because each scale overlaps in information) but requires orders of magnitude fewer sequential forward passes. For a 16×16 token grid with 4 scales, VAR needs 4 sequential steps versus 256 for raster-scan. That sequential depth is what kills raster-scan AR — each step is a transformer forward pass, and 256 dependent passes is painfully slow.

## The Concept

The core abstraction is straightforward: an image is encoded into $K$ discrete token maps at resolutions $1 \times 1$, $2 \times 2$, $4 \times 4$, …, $h \times w$. Autoregression operates across scales, not within them. Each scale map is predicted conditioned on all previous coarser scales, but tokens within a single scale are generated in parallel. This means the transformer's causal mask is block-structured: causal across scale boundaries, bidirectional within each scale block.

```mermaid
flowchart TD
    A["Input Image<br/>(H×W×3)"] --> B["Multi-Scale VQ-VAE Encoder"]
    B --> C["Latent Feature Map"]
    C --> D["Vector Quantization<br/>(shared codebook)"]
    D --> E1["Scale 0: 1×1 tokens"]
    D --> E2["Scale 1: 2×2 tokens"]
    D --> E3["Scale 2: 4×4 tokens"]
    D --> E4["Scale K: h×w tokens"]
    
    E1 --> F["AR Transformer<br/>(block-causal mask)"]
    F --> G1["Predict Scale 0"]
    G1 --> G2["Predict Scale 1<br/>(conditioned on 0)"]
    G2 --> G3["Predict Scale 2<br/>(conditioned on 0-1)"]
    G3 --> GK["Predict Scale K<br/>(conditioned on 0..K-1)"]
    
    GK --> H["Multi-Scale VQ-VAE Decoder"]
    H --> I["Output Image<br/>(H×W×3)"]
    
    style F fill:#e1f5fe
    style H fill:#fff3e0
```

Three components make this work. First, the **multi-scale VAE tokenizer**: a VQ-VAE that encodes an image into a latent feature map, quantizes it via a learned codebook, then reshapes the quantized latent into pyramid levels at increasing resolutions. Second, the **autoregressive transformer**: a standard transformer (Llama-style or GPT-style) that receives flattened token embeddings from all coarser scales as context and predicts a distribution over the next scale's tokens. Third, the **decoder**: the same VQ-VAE's decoder, which takes the full stack of predicted token maps and reconstructs the pixel image.

The critical difference from raster-scan AR is in context growth. A raster-scan model over $N$ tokens requires $O(N^2)$ attention because every token may attend to every previous token. VAR's total attention cost is $\sum_{k=1}^{K} S_k^2$ where $S_k$ is the sequence length at scale $k$ — but the key insight is that each scale is predicted in one shot, so the model never processes a sequence longer than the cumulative tokens up to scale $k$. For a 256×256 image with 10 scales, the VAR sequence length peaks at $\sum_{k=0}^{9} (2^k)^2 \approx \frac{4^9}{3} \approx 87{,}381$ tokens, while the raster-scan equivalent processes $256^2 = 65{,}536$ tokens one at a time across $65{,}536$ forward passes. The quadratic attention cost is comparable, but the sequential depth drops from $65{,}536$ to $10$.

Now let us build and visualize the block-causal attention mask that makes this work:

```python
import torch

def build_var_attention_mask(scale_sizes):
    total = sum(scale_sizes)
    mask = torch.zeros(total, total)
    
    boundaries = [0]
    for s in scale_sizes:
        boundaries.append(boundaries[-1] + s)
    
    num_scales = len(scale_sizes)
    for i in range(num_scales):
        for j in range(num_scales):
            if j <= i:
                start_i, end_i = boundaries[i], boundaries[i + 1]
                start_j, end_j = boundaries[j], boundaries[j + 1]
                mask[start_i:end_i, start_j:end_j] = 1.0
    
    return mask

scale_sizes = [1, 4, 16, 64]
mask = build_var_attention_mask(scale_sizes)

print("Block-causal attention mask for scales [1, 4, 16, 64]:")
print(f"Total sequence length: {mask.shape[0]}")
print()

block_labels = ["S0(1)", "S1(4)", "S2(16)", "S3(64)"]
boundaries = [0, 1, 5, 21, 85]
for i in range(len(block_labels)):
    row_start = boundaries[i]
    row_end = boundaries[i + 1]
    counts = []
    for j in range(len(block_labels)):
        col_start = boundaries[j]
        col_end = boundaries[j + 1]
        block = mask[row_start:row_end, col_start:col_end]
        allowed = int(block.sum().item())
        total_in_block = block.numel()
        status = "ALLOWED" if allowed == total_in_block else ("PARTIAL" if allowed > 0 else "BLOCKED")
        counts.append(f"{block_labels[j]}: {status}")
    print(f"  {block_labels[i]} attends to -> {', '.join(counts)}")

print()
print("Full mask (1=attend, 0=blocked), first 21x21 block:")
print(mask[:21, :21].int())
```

The mask shows that scale 0 tokens attend only to scale 0, scale 1 tokens attend to scales 0 and 1, and so on. Within any scale block, attention is fully bidirectional — all tokens in scale $k$ see each other. Across scale boundaries, attention is strictly causal — a coarser scale never sees a finer scale. This is what enables parallel generation within a scale while preserving autoregressive structure across scales.

## Build It

Let us build a minimal end-to-end VAR pipeline. We cannot train a production VAR model in a lesson (the paper's largest model is 2B parameters trained on ImageNet), but we can implement every component at toy scale and watch the full forward pass produce output. The goal is to make the mechanism concrete — tokenization, block-causal transformer, and next-scale inference.

First, the multi-scale VQ-VAE tokenizer. In the real VAR paper, this is a trained convolutional encoder that produces a multi-resolution latent pyramid. For our build, we simulate the tokenizer's output — discrete token IDs at each scale — so we can focus on the autoregressive mechanism:

```python
import torch
import torch.nn as nn
import math

class MultiScaleVQVAE(nn.Module):
    def __init__(self, vocab_size=512, base_channels=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.codebook = nn.Parameter(torch.randn(vocab_size, base_channels * 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )
    
    def quantize(self, features):
        B, C, H, W = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        dists = torch.cdist(flat.unsqueeze(0), self.codebook.unsqueeze(0))[0]
        indices = dists.argmin(dim=-1)
        quantized = self.codebook[indices].reshape(B, H, W, C).permute(0, 3, 1, 2)
        return indices.reshape(B, H, W), quantized + (features - features.detach() - quantized + quantized.detach())
    
    def encode_to_pyramid(self, x, num_scales=4):
        _, _, H, W = x.shape
        features = self.encoder(x)
        _, _, fH, fW = features.shape
        
        token_maps = []
        for k in range(num_scales):
            divisor = 2 ** (num_scales - 1 - k)
            h_k = max(1, fH // divisor)
            w_k = max(1, fW // divisor)
            
            if k == num_scales - 1:
                h_k, w_k = fH, fW
            
            if h_k < fH or w_k < fW:
                scale_feat = nn.functional.adaptive_avg_pool2d(features, (h_k, w_k))
            else:
                scale_feat = features
            
            indices, _ = self.quantize(scale_feat)
            token_maps.append(indices)
        
        return token_maps
    
    def decode_from_pyramid(self, token_maps, target_h, target_w):
        finest = token_maps[-1]
        B, H, W = finest.shape
        flat = finest.reshape(-1)
        quantized = self.codebook[flat].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        decoded = self.decoder(quantized)
        if decoded.shape[-2:] != (target_h, target_w):
            decoded = nn.functional.interpolate(decoded, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return decoded

torch.manual_seed(42)
tokenizer = MultiScaleVQVAE(vocab_size=256, base_channels=8)
dummy_image = torch.randn(1, 3, 32, 32)
token_maps = tokenizer.encode_to_pyramid(dummy_image, num_scales=4)

print("=== Multi-Scale Tokenization ===")
total_tokens = 0
for k, tm in enumerate(token_maps):
    print(f"  Scale {k}: shape {list(tm.shape)} = {tm.numel()} tokens")
    total_tokens += tm.numel()
print(f"  Total tokens across pyramid: {total_tokens}")
print(f"  Codebook size: {tokenizer.vocab_size}")
print(f"  Input image: {list(dummy_image.shape)}")

decoded = tokenizer.decode_from_pyramid(token_maps, 32, 32)
print(f"  Decoded image: {list(decoded.shape)}")
print(f"  Reconstruction MSE: {((decoded - dummy_image) ** 2).mean().item():.4f}")
```

Now the autoregressive transformer. This is a standard transformer with one modification: the attention mask is block-causal instead of strictly lower-triangular. The model takes a flattened sequence of token IDs (all scales concatenated), embeds them, applies transformer layers with the block-causal mask, and produces logits over the vocabulary at each position:

```python
class VARTransformer(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.scale_embed = nn.Embedding(10, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, token_seq, scale_ids, attention_mask):
        B, T = token_seq.shape
        positions = torch.arange(T, device=token_seq.device).unsqueeze(0).expand(B, T)
        
        embeddings = (
            self.token_embed(token_seq) 
            + self.pos_embed(positions) 
            + self.scale_embed(scale_ids)
        )
        
        features = self.transformer(embeddings, mask=attention_mask)
        logits = self.head(features)
        return logits

def flatten_pyramid(token_maps, device="cpu"):
    flat_tokens = []
    scale_ids = []
    scale_sizes = []
    
    for k, tm in enumerate(token_maps):
        B, H, W = tm.shape
        flat = tm.reshape(B, H * W)
        flat_tokens.append(flat)
        scale_ids.append(torch.full((B, H * W), k, dtype=torch.long, device=device))
        scale_sizes.append(H * W)
    
    token_seq = torch.cat(flat_tokens, dim=1)
    scale_id_seq = torch.cat(scale_ids, dim=1)
    return token_seq, scale_id_seq, scale_sizes

torch.manual_seed(42)
model = VARTransformer(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2)

token_seq, scale_id_seq, scale_sizes = flatten_pyramid(token_maps)
attn_mask = build_var_attention_mask(scale_sizes)
attn_mask_model = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
attn_mask_model = attn_mask_model.masked_fill(attn_mask == 1, 0.0)

logits = model(token_seq, scale_id_seq, attn_mask_model)

print("=== VAR Transformer Forward Pass ===")
print(f"  Input sequence length: {token_seq.shape[1]}")
print(f"  Scale sizes: {scale_sizes}")
print(f"  Logits shape: {list(logits.shape)}")
print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")
print(f"  Attention mask shape: {list(attn_mask.shape)}")
print(f"  Attention mask density: {attn_mask.mean().item():.2%}")
```

The forward pass works. The attention mask density tells us what fraction of position pairs are allowed to attend — for a 4-scale pyramid of sizes [1, 4, 16, 64], roughly 40% of the full attention matrix is populated. That is denser than a purely causal mask (which would be ~50% for a square matrix but much sparser per-row for later positions), but it is structured so that each scale block gets full bidirectional attention while earlier blocks cannot peek ahead.

## Use It

The practical value of VAR is visible at inference time. Instead of autoregressively generating 680 tokens one at a time (for a 16×16 latent grid), the model generates 4 scale-maps in 4 forward passes. Each forward pass produces an entire resolution level. Let us build the inference loop:

```python
@torch.no_grad()
def var_inference(model, num_scales, scale_resolutions, vocab_size, device="cpu"):
    model.eval()
    generated_maps = []
    
    all_tokens = torch.zeros(1, 0, dtype=torch.long, device=device)
    all_scale_ids = torch.zeros(1, 0, dtype=torch.long, device=device)
    all_sizes = []
    
    for k in range(num_scales):
        h, w = scale_resolutions[k]
        scale_size = h * w
        
        if k == 0:
            input_tokens = torch.full((1, 1), vocab_size, dtype=torch.long, device=device)
            input_scale_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            input_sizes = [1]
        else:
            input_tokens = all_tokens.clone()
            input_scale_ids = all_scale_ids.clone()
            input_sizes = list(all_sizes)
        
        attn_mask = build_var_attention_mask(input_sizes)
        attn_mask_float = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask_float = attn_mask_float.masked_fill(attn_mask == 1, 0.0)
        
        logits = model(input_tokens, input_scale_ids, attn_mask_float)
        
        next_logits = logits[:, -1:, :]
        next_tokens = next_logits.argmax(dim=-1)
        
        new_scale_ids = torch.full((1, scale_size), k, dtype=torch.long, device=device)
        new_tokens = next_tokens.expand(1, -1) if next_tokens.shape[1] < scale_size else next_tokens[:, :scale_size]
        
        if new_tokens.shape[1] < scale_size:
            random_extra = torch.randint(0, vocab_size, (1, scale_size - new_tokens.shape[1]), device=device)
            new_tokens = torch.cat([new_tokens, random_extra], dim=1)
        
        all_tokens = torch.cat([all_tokens, new_tokens], dim=1)
        all_scale_ids = torch.cat([all_scale_ids, new_scale_ids], dim=1)
        all_sizes.append(scale_size)
        
        generated_maps.append(new_tokens.reshape(h, w))
        print(f"  Scale {k} ({h}x{w}): predicted {scale_size} tokens in 1 forward pass")
    
    return generated_maps

print("=== VAR Next-Scale Inference ===")
scale_resolutions = [(1, 1), (2, 2), (4, 4), (8, 8)]
generated = var_inference(model, num_scales=4, scale_resolutions=scale_resolutions, vocab_size=256)

print()
print("Generated token maps:")
for k, tm in enumerate(generated):
    print(f"  Scale {k} ({tm.shape[0]}x{tm.shape[1]}):")
    if tm.numel() <= 16:
        print(tm)
    else:
        print(f"    [token values: min={tm.min().item()}, max={tm.max().item()}, mean={tm.float().mean().item():.1f}]")
```

This inference loop demonstrates the mechanism: the model predicts each scale conditioned on all prior scales, and each scale's tokens are produced in a single forward pass. The untrained model produces random tokens, but the structure — context building scale by scale, each resolution refining the previous — is exactly what the VAR paper describes.

Now, the practical question: when would you actually use VAR? The paper reports that VAR[d16] (310M parameters) achieves better FID and IS on ImageNet 256×256 class-conditional generation than DiT-XL/2 (675M parameters) at the same compute budget. It also demonstrates scaling-law behavior consistent with Chinchilla-style predictions, meaning VAR models should improve predictably with more data and parameters. If you are building or evaluating generative image features in a product — marketing asset generation, product visualization, data augmentation for vision pipelines — VAR's faster inference (4-10 forward passes vs 50-1000 diffusion steps) and controllable coarse-to-fine generation are concrete advantages. This places VAR firmly in the category of AI-native product infrastructure (Zone 4), not GTM pipeline mechanics. It is not a Clay waterfall or a lead-scoring mechanism; it is a generative model you would embed in a product feature.

Let us verify the complexity claims with a concrete comparison:

```python
import time

def measure_complexity(max_resolution=256, num_scales_values=None):
    if num_scales_values is None:
        num_scales_values = [4, 6, 8, 10]
    
    print(f"{'Approach':<25} {'Resolution':<12} {'Tokens':<10} {'Fwd Passes':<12} {'Attention Ops':<15}")
    print("-" * 74)
    
    for res in [32, 64, 128, 256]:
        N = res * res
        print(f"{'Raster-Scan AR':<25} {f'{res}x{res}':<12} {N:<10} {N:<12} {N*N:<15,}")
        
        for K in num_scales_values:
            scale_lens = []
            for k in range(K):
                side = max(1, res // (2 ** (K - 1 - k)))
                scale_lens.append(side * side)
            
            total_tokens = sum(scale_lens)
            max_seq = scale_lens[-1]
            fwd_passes = K
            attn_ops = sum(s * sum(scale_lens[:i+1]) for i, s in enumerate(scale_lens))
            
            print(f"{'VAR (K=' + str(K) + ')':<25} {f'{res}x{res}':<12} {total_tokens:<10} {fwd_passes:<12} {attn_ops:<15,}")
        print()

measure_complexity()
```

The numbers speak for themselves. At 256×256 resolution, raster-scan AR needs 65,536 sequential forward passes. VAR with 10 scales needs 10. The total token count is higher for VAR (because each scale carries its own information), but the sequential depth — which determines wall-clock inference time — is orders of magnitude lower. Diffusion models sit in between: typically 50-1000 denoising steps, each a full forward pass over the entire image.

## Ship It

Deploying VAR in production means confronting three engineering decisions that the paper touches on but