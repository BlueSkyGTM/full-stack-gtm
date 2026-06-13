# Music Generation — MusicGen, Stable Audio, Suno, and the Licensing Earthquake

## Beat 1: Hook

The sync licensing market is worth ~$500M annually. AI-generated music just broke the jukebox — and the legal system has no idea what to do about it. Three architectures now compete to generate full tracks from text prompts: autoregressive token prediction (MusicGen), latent diffusion over spectrograms (Stable Audio), and a closed composite pipeline that outputs radio-ready songs with vocals (Suno). Each carries different licensing baggage. This lesson maps the mechanisms, the outputs, and the liability.

## Beat 2: Concept

**Mechanism 1 — Autoregressive Audio Token Prediction (MusicGen):** Audio is compressed into discrete tokens using EnCodec, then predicted sequentially by a transformer. The model generates one token at a time, conditioned on text and optional melody reference. Training data: licensed music (Meta's internal dataset). Weights are open. Output: instrumental only, short clips (up to 30s), artifacts common at segment boundaries.

**Mechanism 2 — Latent Diffusion on Time-Frequency Representations (Stable Audio):** A variational autoencoder compresses mel-spectrograms into a latent space. A diffusion model learns to denoise within that space, conditioned on text embeddings and timing. Training data: licensed via Audiosparx partnership. Weights are open. Output: instrumental, longer form (up to ~47s at 44.1kHz), better temporal coherence than autoregressive.

**Mechanism 3 — Closed Composite Pipeline (Suno):** Architecture is undocumented. Observable behavior: generates full songs with vocals, lyrics, verse-chorus structure, genre-appropriate instrumentation. Claims no user-facing model weights. Training data provenance: undisclosed. Output: radio-quality full tracks. This is the one the labels are suing over.

**The Licensing Earthquake:** Three rights bundles collide — composition rights (melody/harmony/lyrics), sound recording rights (the specific audio waveform), and performance rights (who performed it). AI models trained on copyrighted recordings may infringe on sound recording rights even if the output is novel. Suno and Udio face active lawsuits from RIAA. MusicGen and Stable Audio trained on licensed data but output may still resemble protected works. The mechanism matters: diffusion models memorize less than autoregressive models, which memorize less than retrieval systems. [CITATION NEEDED — concept: quantified memorization rates by music generation architecture type]

## Beat 3: Demo

Local inference with MusicGen via the `audiocraft` library — generate a 10-second clip from a text prompt, save to WAV, and inspect the EnCodec token sequence to observe the autoregressive bottleneck. Second demo: API call to Suno via `suno-api` wrapper to generate a full song with vocals, then compare file size, duration, and spectrogram characteristics against the MusicGen output. Observable output: saved WAV files, printed token counts, and spectrogram PNG.

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

model = MusicGen.get_pretrained('small')
model.set_generation_params(duration=10)
wav = model.generate(['ambient electronic pad with slow arpeggios'], progress=True)
for idx, one_wav in enumerate(wav):
    audio_write(f'musicgen_output_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    print(f"Generated clip {idx}: shape={one_wav.shape}, sample_rate={model.sample_rate}")
    print(f"File saved: musicgen_output_{idx}.wav")

waveform, sr = torchaudio.load(f'musicgen_output_0.wav')
spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128)
spec = spec_transform(waveform)
plt.figure(figsize=(10, 4))
plt.imshow(np.log(spec[0].numpy() + 1e-9), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Log Magnitude')
plt.title('MusicGen Output Mel Spectrogram')
plt.xlabel('Time Frame')
plt.ylabel('Mel Band')
plt.tight_layout()
plt.savefig('musicgen_spectrogram.png', dpi=150)
print("Spectrogram saved: musicgen_spectrogram.png")
```

## Beat 4: Use It

**GTM Cluster: Content Operations — Brand Audio & Podcast Production**

Mechanism-to-application mapping: sales teams record podcast intros, product demo walkthroughs, and social clips. Sync licensing for 15-second background beds costs $50–500 per use via standard libraries. MusicGen and Stable Audio produce royalty-free instrumental beds at zero marginal cost — but "royalty-free" does not mean "lawsuit-free." The practitioner must verify the model's training data license before shipping output in commercial GTM assets.

**GTM Redirect:** This is the content production layer in Zone 2 (Engagement). Specifically: AI-generated audio for video thumbnails, podcast bumpers, and demo background music in outbound sequences. The Clay waterfall personalizes the text layer; music generation personalizes the audio layer. The licensing constraint is the blocking dependency — the practitioner cannot ship audio to prospects without clear commercial rights.

**Decision framework for GTM use:**
- MusicGen (open weights, licensed training data): safe for internal demos, marginal risk for client-facing assets
- Stable Audio (open weights, Audiosparx license): check the Stability AI license terms for commercial use
- Suno (closed, undisclosed training data): highest output quality, highest legal risk for commercial GTM use

## Beat 5: Ship It

**Easy:** Generate three 10-second instrumental beds using MusicGen with different genre prompts. Save to disk and compare file sizes and spectrogram density.

**Medium:** Build a CLI tool that accepts a mood descriptor and duration, generates a backing track via MusicGen, and outputs licensing metadata (model version, training data license, generation timestamp) in a JSON sidecar file.

**Hard:** Construct a spectrogram-based similarity pipeline: generate 5 clips via MusicGen, compute cosine similarity between their mel-spectrogram embeddings, and flag any pair exceeding 0.85 similarity as a potential memorization/infringement risk. Output a similarity matrix PNG and a list of flagged pairs.

## Beat 6: Quiz

Questions grounded in mechanisms and observable behavior:

1. MusicGen uses EnCodec to compress audio into discrete tokens before autoregressive generation. What artifact does this tokenization introduce at segment boundaries in clips longer than 15 seconds?
2. Stable Audio uses latent diffusion over mel-spectrogram representations. Explain why this mechanism produces better temporal coherence than autoregressive token prediction for longer clips.
3. A GTM team wants to use Suno-generated songs in a client-facing video campaign. Name three rights bundles that must be verified before shipping, and explain why Suno's undisclosed training data creates the highest legal exposure among the three tools covered.
4. Compare the memorization risk profile: diffusion-based audio generation (Stable Audio) vs. autoregressive token prediction (MusicGen). Which architecture is more likely to reproduce near-duplicate fragments of training recordings, and why?
5. You generate a 30-second MusicGen clip and observe a cosine similarity of 0.92 against a known copyrighted recording's spectrogram. Describe the next two steps you would take before deciding whether to ship this audio in a commercial asset.

---

**GTM Redirect Summary:** Foundational for Zone 2 (Content Operations). Direct application to brand audio production, podcast scoring, and demo background music. Blocking dependency: licensing verification before any client-facing deployment.