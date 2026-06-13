# CNNs — LeNet to ResNet

## GTM Redirect Rules
- Primary GTM cluster: Zone 1 — Signal Capture (visual firmographics, logo detection, screenshot-based company intelligence)
- Secondary: Zone 2 — Enrichment (OCR for document intake, visual data extraction)
- If no clean application exists: "foundational for any pipeline processing image data"

---

## Beat 1: Hook — Why Depth Matters

The ImageNet moment: 2012's AlexNet cut error rates by 40% overnight. The mechanism wasn't more data — it was depth done right. Describe the empirical trajectory from LeNet's 5 layers to ResNet's 152, and the vanishing gradient problem that made each jump non-obvious.

---

## Beat 2: Concept — Architectural Mechanisms in Sequence

**LeNet (1998):** Conv → Pool → Conv → Pool → FC. The basic template. Introduce receptive fields and weight sharing as the core constraint-relaxation vs. fully-connected nets.

**AlexNet (2012):** ReLU activation (sparse gradients, faster convergence), dropout (ensemble approximation via random ablation), GPU parallelism. Mechanism: why ReLU avoids vanishing gradients that sigmoid produces.

**VGG (2014):** Stacked 3×3 filters replacing larger kernels. Mechanism: two 3×3 convolutions have the same receptive field as one 5×5, with fewer parameters and more nonlinearity injections.

**GoogLeNet/Inception (2014):** Parallel filter banks at multiple scales concatenated. Mechanism: letting the network learn which receptive field size matters rather than imposing one.

**ResNet (2015):** Skip connections. Mechanism: identity mapping as default, residual as learned delta. Gradient flows through addition, not multiplication — the mathematical reason 152 layers becomes trainable.

---

## Beat 3: Demonstration — Implementing the Key Innovations

Working code showing:
1. A convolution operation from scratch (no library) — print input tensor, kernel, output, confirm receptive field
2. ReLU vs. sigmoid gradient flow at depth — demonstrate dead neurons empirically
3. A residual block in PyTorch — show gradient magnitude with and without skip connection across 50 layers

**Exercise hooks:**
- Easy: Modify kernel size in the from-scratch conv and observe output dimension changes
- Medium: Stack 20 conv layers with sigmoid vs. ReLU, print gradient norms at each layer
- Hard: Build a 34-layer ResNet and a 34-layer plain net on CIFAR-10, plot training curves side-by-side

---

## Beat 4: Use It — Feature Extraction for GTM Signals

**GTM Redirect:** Zone 1 — Signal Capture. Visual firmographic intelligence: detecting logos on company websites, classifying screenshot content, extracting visual intent signals from landing pages.

Use a pre-trained ResNet (torchvision) as a frozen feature extractor. Run company screenshots through it, extract 2048-d embedding, cluster similar visual designs. This is the same pipeline used for visual competitive intelligence — detecting when prospects redesign, identifying shared design patterns across a vertical.

**Exercise hooks:**
- Easy: Load pre-trained ResNet, strip the classification head, print embedding shape for a single image
- Medium: Extract embeddings for a directory of screenshots, compute cosine similarity matrix, print top-5 nearest neighbors
- Hard: Build a logo presence classifier using ResNet features + a linear probe on a labeled dataset of 200 company pages

---

## Beat 5: Ship It — Production CNN Pipeline

Architectural decisions for deployment: input size constraints (224×224 standard), batch normalization behavior at inference, model size vs. latency tradeoffs (ResNet-18 vs. ResNet-152). Export to TorchScript, measure inference latency, log embedding drift over time to detect when the visual distribution shifts.

**Exercise hooks:**
- Easy: Script a ResNet-18, time 100 inferences, print p50/p95 latency
- Medium: Build a batch embedding pipeline that processes a directory of images and writes embeddings to a .parquet file with metadata
- Hard: Implement embedding drift detection: compare today's batch embeddings to last week's centroid using Mahalanobis distance, flag if above threshold

---

## Beat 6: Evaluate — Architectural Reasoning

Fluency check: given a new architecture description, the practitioner can predict which historical problem it solves and trace the mechanism.

**Exercise hooks:**
- Easy: Given a 4-layer CNN diagram, label each layer's function (feature extraction, downsampling, classification)
- Medium: Given a "mystery" architecture description with skip connections every 2 layers and 1×1 convs for dimension matching, identify it as a ResNet variant and explain why the 1×1 convs exist
- Hard: Read the DenseNet paper's connectivity pattern (dense blocks with concatenation), compare gradient flow mechanics to ResNet's addition-based skips, and write a 3-sentence tradeoff analysis

---

## Learning Objectives

1. Implement a convolution operation from scratch and trace how receptive fields build across layers.
2. Compare LeNet, AlexNet, VGG, and ResNet architectures by the specific degradation problem each solved.
3. Build a residual block and demonstrate gradient flow improvement over a plain network at depth >20 layers.
4. Extract embeddings from a pre-trained ResNet and compute similarity across a set of images.
5. Deploy a frozen CNN feature extractor as a TorchScript module with latency measurement and drift logging.