# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split

---

## Beat 1: Hook

The bottleneck: raw audio is 24,000+ samples per second. Language models need discrete tokens. Neural audio codecs bridge this gap by compressing continuous waveforms into finite token sequences at ~50 frames per second — with reconstruction quality high enough that humans can't tell the difference. Every modern speech model (Voicebox, Moshi, AudioCraft) runs on codec tokens. If you're building voice AI, you're building on top of one of these four architectures whether you know it or not.

---

## Beat 2: Concept

**Residual Vector Quantization (RVQ)** — the core mechanism. An encoder compresses audio into a continuous latent. A cascade of N quantizers each encode the *residual error* from the previous stage. More codebooks = higher fidelity, more tokens to generate. EnCodec (Meta, 2022) established this pattern with 8 codebooks at 75 Hz. DAC (Descript, 2023) improved it with improved residual vector quantization, larger codebooks, and snake activations. SNAC (Kyuutai, 2024) introduced *multi-scale* quantization — codebooks operate at different temporal resolutions (75 Hz, 40 Hz, 25 Hz), capturing coarse structure at low frame rates and fine detail at high frame rates. Mimi (Kyuutai, 2024, powers Moshi) splits the codec into a *semantic* codebook (distilled from a pretrained speech encoder like HuBERT/wav2vec2) and *acoustic* codebooks (standard RVQ). The semantic-acoustic split: semantic tokens capture "what was said" (phonetic, linguistic content), acoustic tokens capture "how it sounded" (speaker identity, room acoustics, emotion). Downstream models can generate semantic tokens autoregressively (easier, linguistic structure) and predict acoustic tokens in parallel (harder, fine detail). [CITATION NEEDED — concept: Mimi semantic distillation training procedure from HuBERT]

---

## Beat 3: Demo

Tokenize a short audio clip with EnCodec (via `encodec` library). Inspect the shape of the output tensor: `(batch, num_codebooks, frames)`. Decode it back. Measure reconstruction error with simple MSE. Then vary the number of codebooks used at decode time (8 → 4 → 2 → 1) and listen to or measure the quality degradation — this demonstrates the RVQ residual hierarchy directly. Observable output: print codebook shapes, MSE at each codebook count, and a spectrogram comparison.

**Exercise hook (easy):** Load a WAV file, tokenize it with EnCodec, decode with all codebooks, and print the shape of the token tensor plus MSE between original and reconstructed audio.

---

## Beat 4: Lab

Implement a side-by-side comparison of EnCodec and DAC (via `descript-audio-codec` package). For the same audio clip, tokenize with both, compare: token rate (Hz), number of codebooks, vocabulary size per codebook, and reconstruction quality. Then implement the semantic-acoustic split conceptually: extract wav2vec2 features from the same audio, compute cosine similarity between wav2vec2 frame-level embeddings and each EnCodec codebook's embedding table. You'll see which codebook is most "semantic" (highest average cosine similarity with wav2vec2). Observable output: printed tables of codec specs and similarity scores per codebook.

**Exercise hook (medium):** For each of EnCodec's 8 codebooks, compute the average cosine similarity between that codebook's learned embeddings and wav2vec2 features extracted from the same audio. Plot a bar chart showing which codebook is most semantically aligned.

**Exercise hook (hard):** Implement the multi-scale tokenization pattern from SNAC conceptually: take EnCodec codebooks and group them into 3 temporal scales by downsampling frames at rates [1, 2, 3]. Reconstruct by upsampling back and summing contributions. Compare reconstruction quality to standard RVQ decode.

---

## Beat 5: Use It

**GTM Redirect: Zone 30 — AI-Powered Conversation & Voice Agents.**

Neural audio codecs are the tokenization layer for every real-time voice AI system. When you're configuring a voice agent pipeline (e.g., a Clay waterfall for lead qualification over phone), the codec choice determines: latency (Mimi was designed for streaming at 80ms latency), voice quality (DAC > EnCodec at equivalent bitrate), and speaker consistency (semantic tokens stabilize identity across turns). If you're evaluating or building voice agent infrastructure, you need to evaluate codec specs — frame rate, codebook count, streaming support — not just the language model on top. Foundational for any pipeline that converts speech to tokens and back.

**Exercise hook (medium):** Design a latency budget for a voice agent that uses Mimi for tokenization: encode (X ms), LLM inference (Y ms), decode (Z ms). Given Mimi's documented 80ms frame size and 8 codebooks, calculate the minimum achievable round-trip latency and identify the bottleneck.

---

## Beat 6: Ship It

Build a CLI tool that takes an audio file, tokenizes it with EnCodec at N codebooks (user-configurable), saves the discrete tokens to disk as numpy arrays, and reconstructs audio from those tokens. The tool prints: original file size, token array size, compression ratio, and reconstruction MSE. This is the minimal building block for any codec-based audio pipeline — storage, transmission, or generation.

**Exercise hook (hard):** Extend the CLI tool to support multiple codecs (EnCodec, DAC). Add a `--compare` flag that tokenizes with both, prints side-by-side metrics (token count, compression ratio, MSE, RT60 of reverb preservation), and writes both reconstructions to disk for blind A/B comparison.

---

## Learning Objectives

1. **Compare** RVQ-based codec architectures (EnCodec, DAC) against multi-scale approaches (SNAC) on dimensions of frame rate, codebook count, and reconstruction fidelity.
2. **Explain** the semantic-acoustic token split and its functional role in separating linguistic content from speaker/room characteristics.
3. **Implement** audio tokenization and detokenization with a neural codec, producing discrete token tensors and measuring reconstruction error.
4. **Evaluate** which codebook in an RVQ codec carries the most semantic information by computing similarity against a pretrained speech encoder (wav2vec2).
5. **Configure** a codec-based audio pipeline with appropriate latency and quality tradeoffs for a target application (streaming voice agent vs. offline generation).