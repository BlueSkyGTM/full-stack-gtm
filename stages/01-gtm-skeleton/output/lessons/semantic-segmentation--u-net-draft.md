# Semantic Segmentation — U-Net

## Hook
Pixel-level classification — every pixel gets a label, not just the whole image. This is what makes "where is the thing?" tractable, and it's the architectural ancestor of every modern encoder-decoder model with skip connections.

## Concept
The U-Net mechanism: an encoder path compresses spatial resolution through repeated convolution + pooling, a decoder path recovers it through transposed convolutions, and skip connections concatenate encoder feature maps to decoder layers at matching resolutions. The skip connection is the critical innovation — it re-injects spatial detail that the bottleneck would otherwise destroy.

## Demo
Build a minimal U-Net in PyTorch. Run a forward pass on a synthetic 128×128 tensor. Print shapes at every encoder stage (128→64→32→16→8), every decoder stage (8→16→32→64→128), and at each skip concatenation to make the information flow visible.

## Use It
The encoder-decoder pattern with skip connections is foundational for **Zone 06 — Embeddings, semantic search**. The same architectural principle (compress to a representation, reconstruct with retained context) underpins embedding models used in inbound signal routing. The GTM redirect: this is how embedding models compress signals, not a direct Clay waterfall application.

**Exercise hooks:**
- **Easy:** Given a U-Net with 4 encoder blocks, predict the tensor shape after each pooling operation.
- **Medium:** Replace max-pooling with strided convolution; compare parameter counts and run a forward pass to confirm identical output shapes.
- **Hard:** Train the minimal U-Net on a synthetic binary segmentation task (random circles on noise); print per-epoch IoU.

## Ship It
Wrap U-Net inference into a callable function that accepts a batch of images, returns per-pixel class probabilities, and thresholds to a segmentation mask. Export via `torch.jit.script` and verify the scripted model produces identical output to the eager model.

**Exercise hooks:**
- **Easy:** Script the trained model and confirm matching outputs between eager and TorchScript.
- **Medium:** Add a preprocessing function (normalize, resize) and package it with the model into a single inference class.
- **Hard:** Benchmark inference latency at batch sizes 1, 4, 16; print throughput (images/sec) for each.

## Evaluate
Quiz hooks derived from mechanisms demonstrated in the Demo:
1. What spatial resolution change does each max-pool operation cause, and why can't the decoder recover this without skip connections?
2. Predict the channel dimension after a skip concatenation given encoder output channels = 64 and decoder output channels = 32.
3. Explain why U-Net uses concatenation (not addition) for skip connections — what information does this preserve that addition would lose?
4. Given input 256×256 and 3 encoder blocks, trace the spatial dimensions through the full U-shape.