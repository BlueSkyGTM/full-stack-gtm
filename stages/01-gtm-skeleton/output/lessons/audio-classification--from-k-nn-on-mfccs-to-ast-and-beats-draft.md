# Audio Classification — From k-NN on MFCCs to AST and BEATs

## Learning Objectives

1. Extract MFCC features from audio signals and explain their role as compressed spectral representations.
2. Implement k-NN classification on MFCC vectors and evaluate baseline accuracy.
3. Compare spectrogram-based transformer architectures (AST) against traditional feature pipelines.
4. Run zero-shot and fine-tuned audio classification using BEATs.
5. Diagnose when a simple MFCC+classifier pipeline outperforms a transformer and articulate the tradeoff.

---

## Beat 1: Hook

The 40-year-old pipeline (extract features → classify) and the 3-year-old pipeline (raw waveform → transformer) both solve the same problem. The gap between them is your design space. We trace one signal through both.

---

## Beat 2: Concept (Mechanism First)

**MFCCs as dimensionality reduction.** The cochlea filters sound into frequency bands; MFCCs approximate this with a discrete cosine transform on mel-scaled power spectra. You go from ~20kHz waveform to 13–40 coefficients per frame. This is the feature space where k-NN actually works.

**Code hook:** Extract MFCCs from a `.wav` file using `librosa`, print shape, plot one frame's coefficient vector.

**Exercise (easy):** Load two audio files from different classes (e.g., speech vs. music), extract MFCCs, print the mean vector for each — observe separation.

---

## Beat 3: Theory (Spectral → Spatial → Attention)

Three representations, three classifiers:

| Representation | Classifier | Inductive bias |
|---|---|---|
| MFCC vectors | k-NN / logistic regression | Distance in cepstral space correlates with class |
| Log-mel spectrogram | CNN | Local time-frequency patterns are discriminative |
| Log-mel spectrogram patches | Transformer (AST) | Any patch can attend to any other patch |

**AST mechanism:** Split spectrogram into 16×16 patches (same as ViT), flatten, add positional embeddings, classify with a `[CLS]` token. No convolution. No recurrence. Pure attention over time-frequency blocks.

**BEATs mechanism:** Self-supervised pre-training via masked audio prediction, then fine-tune a linear classifier on frozen features. The tokenizer learns discrete audio tokens without requiring labels.

**Code hook:** Run `ast-fine-tuned-audioset-10-10-0.4593` from Hugging Face on a sample, print top-5 predictions.

**Exercise (medium):** Implement k-NN on ESC-50MFCCs (use a 5-fold split), report mean accuracy. Then run AST zero-shot on the same files. Compare.

---

## Beat 4: Use It

**GTM redirect:** Conversation intelligence and call-coaching platforms (Zone 02 — Signal Detection). Audio classification determines whether a sales call contains a competitor mention, a pricing objection, or a buying signal. The same architecture classifies podcast mentions of your brand, support call escalation risk, and demo recording quality.

**Mechanism → tool chain:**
1. Audio source (call recording, podcast clip) →
2. Frame-level classification (speech vs. silence vs. music) →
3. Segment-level classification (objection type, sentiment, topic) →
4. Structured signal to CRM / Clay enrichment waterfall

**Code hook:** Classify a directory of `.wav` files into business-relevant categories (or ESC-50 stand-ins) using AST, output a CSV with `file_path, predicted_label, confidence`.

**Exercise (medium):** Build a batch classifier that processes 10 audio files, writes results to CSV, and logs inference time per file. Compare AST vs. BEATs latency.

---

## Beat 5: Ship It

**Production constraints:** Real-time call classification requires <300ms latency per minute of audio. MFCC extraction is ~5ms. AST inference on CPU is ~200ms for a 5-second clip. Your architecture choice is a latency/accuracy tradeoff, not a purity test.

**Ensemble pattern:** Use MFCC+logistic-regression as a cheap router (speech vs. non-speech, silence detection), then pass only relevant segments to AST for fine-grained classification. This is the same pattern as LLM guardrails — fast filter, expensive verify.

**Code hook:** Production inference function: takes audio path, runs MFCC pre-filter, conditionally runs AST, returns structured classification dict. Print full pipeline timing.

**Exercise (hard):** Implement the two-stage pipeline. Measure P95 latency on 100 files. Report accuracy drop (if any) from the MFCC pre-filter rejecting valid segments.

---

## Beat 6: Stretch

**Where BEATs wins and where it doesn't.** BEATs (Bidirectional Encoder representation via Audio Tokens) learns discrete audio tokens self-supervised, then fine-tunes. It outperforms AST on AudioSet when you have enough pre-training data. It underperforms on small, domain-specific datasets where the tokenizer's vocabulary doesn't match your signal.

**Open questions to explore:**
- What happens when you fine-tune BEATs on 50 examples vs. 5000? Where is the floor?
- Can you use BEATs tokens as input to a *separate* classifier (feature extraction mode) and beat the built-in classifier?
- How do you handle variable-length audio (>30s calls) when AST expects fixed-length spectrograms? Chunking introduces boundary artifacts — what's the mitigation?

**Exercise (hard):** Fine-tune BEATs on a 5-class subset of ESC-50 with only 20 training examples per class. Compare against k-NN on MFCCs with the same data. Plot accuracy vs. training set size for both.

---

## GTM Redirect Summary

**Primary cluster:** Zone 02 — Signal Detection (conversation intelligence, call classification, voice-of-customer signal extraction from audio).

**Secondary cluster:** Zone 01 — Enrichment (classifying audio attachments, podcast mentions, voicemail categorization as structured data points on a contact record).

**Foundational note:** If the audio task is generic (environmental sound, music genre), the GTM redirect is foundational. The mechanism transfers — the specific labels don't.

---

## Citation Status

- [CITATION NEEDED — concept: BEATs tokenization vocabulary mismatch on domain-specific audio] — the claim that BEATs underperforms on small domain-specific datasets needs a source.
- AST patch mechanism is documented in *AST: Audio Spectrogram Transformer* (Gong et al., 2021).
- MFCC derivation is standard signal processing; any speech processing textbook covers this.