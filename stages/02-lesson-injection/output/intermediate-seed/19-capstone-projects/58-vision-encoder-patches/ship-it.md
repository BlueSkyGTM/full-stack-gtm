## Ship It

When deploying a vision encoder in an enrichment pipeline, three decisions matter at the patch tokenization layer.

**Input resolution.** Most pretrained ViT models expect 224×224 or 384×384 inputs. If your screenshots are larger, you resize them down before patching. This resize is lossy — a 1920×1080 homepage screenshot compressed to 224×224 loses most of its text legibility. Some pipelines use higher-resolution variants (384×384 or even 512×512) at proportionally higher token counts. The enrichment model's accuracy on fine-grained signals like small logos or footer text depends heavily on this choice.

**Patch size.** If you are fine-tuning or training from scratch, you control the patch size. If you are using a pretrained checkpoint, the patch size is baked in — you cannot change it without reinitializing the projection weights. CLIP's vision encoder uses 32×32 patches on a 224×224 input (49 tokens), which is fast but coarse. ViT-Base uses 16×16 (197 tokens), which is the standard accuracy/speed tradeoff.

**Batching strategy.** Enrichment pipelines process many images. The patch tokenization step is cheap relative to the transformer layers, but the token count it produces determines the memory footprint of every subsequent attention layer. If you batch 32 screenshots at 196 tokens each with hidden dimension 768 and 12 layers, you need to hold intermediate activations for 32 × 197 × 768 × 12 floats — plus the attention matrices. Profile your GPU memory at the patch count you choose before deploying at scale.

```python
import torch
import torch.nn as nn

class PatchTokenizer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, hidden_dim):
        super().__init__()
        assert image_size % patch_size == 0, f"Image size {image_size} not divisible by patch size {patch_size}"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_dim) * 0.02
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        return x

tokenizer = PatchTokenizer(
    image_size=224,
    patch_size=16,
    in_channels=3,
    hidden_dim=768
)

batch = torch.randn(8, 3, 224, 224)
output = tokenizer(batch)
print(f"Batch size:        {batch.shape[0]}")
print(f"Input shape:       {batch.shape}")
print(f"Output shape:      {output.shape}")
print(f"Tokens per image:  {output.shape[1]}")
print(f"Embedding dim:     {output.shape[2]}")
print(f"CLS token[: 5]:    {output[0, 0, :5].detach().tolist()}")

batch_large = torch.randn(8, 3, 384, 384)
tokenizer_384 = PatchTokenizer(384, 16, 3, 768)
output_384 = tokenizer_384(batch_large)
print(f"\n384x384 output:    {output_384.shape}")
print(f"Tokens per image:  {output_384.shape[1]}")
print(f"Cost ratio vs 224: {output_384.shape[1] / output.shape[1]:.1f}x")
```

```
Batch size:        8
Input shape:       torch.Size([8, 3, 224, 224])
Output shape:      torch.Size([8, 197, 768])
Tokens per image:  197
Embedding dim:     768
CLS token[: 5]:    [...]

384x384 output:    torch.Size([8, 577, 768])
Tokens per image:  577
Cost ratio vs 224: 2.9x
```

The `PatchTokenizer` class above is the front end of every standard ViT. It accepts a batch of images, projects patches via `Conv2d`, prepends the CLS token, and adds learned positional embeddings. The assertion on divisibility prevents silent data loss. This is the module you would drop into a custom vision encoder for an enrichment pipeline.