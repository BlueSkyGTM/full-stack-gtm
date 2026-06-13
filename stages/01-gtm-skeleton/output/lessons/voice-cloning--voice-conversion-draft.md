# Voice Cloning & Voice Conversion

## Hook It

A sales team records 200 personalized voicemails per day. Voice cloning reduces this to recording once, then synthesizing the rest. The mechanism is speaker embedding extraction → conditioning a generative model → waveform synthesis. This lesson covers both cloning (generating speech in a target voice from text) and conversion (transforming one voice into another while preserving content).

## Ground It

**Core mechanisms, ordered by dependency:**

1. **Speaker embeddings** — x-vectors and d-vectors compress a voice into a fixed-length vector. This is the "voice fingerprint." Extraction runs through a time-delay neural network trained on speaker verification tasks.

2. **Spectrogram generation** — Text-to-speech models condition on speaker embeddings to produce mel-spectrograms. Tacotron-style attention, or diffusion-based generation (as in NaturalSpeech, Voicebox).

3. **Vocoder synthesis** — Mel-spectrograms become waveforms. HiFi-GAN is the standard; it upsamples mel frames using transposed convolutions with multi-scale discriminators trained adversarially.

4. **Voice conversion vs. cloning distinction** — Conversion preserves the source prosody and timing, swapping only the timbral identity. Cloning generates from text, so prosody is synthesized, not transferred.

Key architectural families:
- **Encoder-decoder TTS** (Tacotron 2, Glow-TTS): attention-based alignment, autoregressive or non-autoregressive
- **Diffusion-based** (DiffWave, Voicebox, NaturalSpeech2): iteratively denoise from Gaussian noise conditioned on speaker embedding
- **Retrieval-based** (RVC - Retrieval-based Voice Conversion): train a shallow model on target voice features, use content features from a source

## Map It

**Tool landscape mapped to mechanism:**

| Tool | Mechanism | Cloning or Conversion | Input |
|------|-----------|----------------------|-------|
| ElevenLabs | Proprietary diffusion model (undocumented architecture) | Cloning (zero/few-shot) | Text + reference audio |
| XTTSv2 (Coqui) | Multi-stage: encoder → GPT-based autoregressive → vocoder | Cloning (few-shot) | Text + reference audio |
| OpenVoice | Tone color transfer via decomposition of voice into timbre + prosody | Conversion + Cloning | Source audio + reference audio |
| RVC (Retrieval-based Voice Conversion) | Feature extraction (HuBERT/ContentVec) → shallow generator | Conversion | Source audio + trained model |
| so-vits-svc | Soft-VC content encoder → VITS decoder | Conversion | Source audio + trained model |

**Trade-offs:**

- **Latency vs. quality**: Autoregressive models (XTTSv2) produce higher naturalness but slower inference than non-autoregressive (Voicebox).
- **Few-shot vs. fine-tuned**: ElevenLabs and XTTSv2 clone from 3-30 seconds of audio. RVC and so-vits-svc require 5-30 minutes of target audio and a fine-tuning step, but achieve higher speaker similarity.
- **Controllability**: OpenVoice allows granular control over emotion, accent, rhythm. Autoregressive models offer less explicit control.

## Build It

**Exercise 1 (Easy): Zero-shot cloning with XTTSv2**
Generate speech from text using a reference audio clip. Print audio file metadata (sample rate, duration) and play the output. Hook: "Clone a voice from a 15-second clip and generate a greeting."

**Exercise 2 (Medium): Voice conversion with OpenVoice**
Take a source recording and convert it to match a target speaker's timbre. Print confirmation of both audio files loading and the converted file path. Hook: "Convert your voice to sound like a reference speaker."

**Exercise 3 (Hard): Speaker similarity evaluation**
Extract speaker embeddings from both the original and cloned audio using resemblyzer or speechbrain. Compute cosine similarity. Print the similarity score and a threshold-based pass/fail judgment. Hook: "Build a quality gate: reject clones below 0.75 cosine similarity."

## Use It

**GTM redirect: Zone 2 — Enrichment & Signal layer.**

Voice cloning maps to the **AI Personalization cluster** in the GTM topic map. Specifically:

- **Personalized video outreach**: Tools like Synthesia and Tavus use voice cloning to generate unique video messages per prospect. The mechanism: CRM data → template script → TTS with cloned sales rep voice → video composite.
- **Localized content at scale**: Record a product demo once. Clone the presenter's voice. Generate narration in multiple languages using voice conversion that preserves the original speaker's identity. [CITATION NEEDED — concept: multilingual voice cloning for sales enablement]
- **Accessibility accommodation**: Voice conversion can transform synthetic TTS into a brand-consistent voice for prospects who prefer audio content.

**Concrete pipeline**: Salesforce contact → Clay waterfall enrichment → first name + company extraction → templated greeting script → XTTSv2 clone of SDR voice → attach .wav to outbound email via SendGrid API.

## Ship It

**Production considerations:**

1. **Consent and licensing**: Cloning a voice without explicit consent is legally actionable in multiple jurisdictions (US right of publicity, EU GDPR biometric data). Store consent records.

2. **Speaker similarity thresholds**: Cosine similarity between source embedding and clone embedding should exceed 0.75 for deployment. Below 0.60, the output sounds like "a similar person, not the same person."

3. **Latency budgets**: Real-time conversion requires <200ms inference. XTTSv2 on CPU: ~3-5 seconds per sentence. ElevenLabs API: ~500ms. OpenVoice on GPU: ~150ms. Profile before promising real-time.

4. **Artifact detection**: Cloned speech contains spectral artifacts above 7kHz. If downstream processing (telephony, compression) filters these, quality degrades. Test with the actual delivery channel.

5. **Monitoring**: Log speaker similarity scores, inference latency, and user-reported quality flags per generation. Alert if similarity drops below threshold for any model update.

**Exercise hook (Hard):** "Build a CI pipeline that rejects a voice model if its cosine similarity against the reference voice drops below 0.75 after any retrain."