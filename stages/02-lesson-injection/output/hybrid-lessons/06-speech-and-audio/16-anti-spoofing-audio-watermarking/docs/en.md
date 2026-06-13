# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify

## Learning Objectives

1. **Classify** an audio clip as bona fide or spoofed using a countermeasure model, and report the detection score relative to a decision threshold.
2. **Embed** a detectable watermark into a synthetic audio waveform using AudioSeal's encoder, and verify its survival after codec degradation.
3. **Compare** the mechanisms of anti-spoofing detection, audio watermarking, and cryptographic provenance signing — and identify which defense applies to which threat model.
4. **Implement** a tampering localization pipeline that reads per-sample watermark presence to identify which audio segments were modified post-generation.
5. **Evaluate** a voice AI deployment against ASVspoof 5 deployment conditions and determine the minimum defense stack required to ship.

## The Problem

Voice cloning needs under three seconds of sample audio to produce a convincing replica. A GTM team deploying conversational outbound — AI SDR calls, voice-modulated product demos, automated follow-up sequences — ships that capability into an environment where the receiver has no mechanism to distinguish an authorized AI voice from an attacker's clone. The attacker records your AI SDR's opening line, feeds three seconds of it into a cloning service, and generates a variant that bypasses any voice-based authentication your downstream systems rely on. This is not hypothetical: voice cloning APIs are publicly available, and the quality sufficient to fool human listeners shipped in 2023.

The trust problem splits into two sub-problems with different threat models. **Spoofing detection** answers the question "is this voice real or synthetic?" — a classification problem where the adversary does not cooperate with your defense. **Watermarking** answers "did this audio originate from our pipeline?" — a provenance problem where you embed a signal at generation time and verify it later. The adversary for watermarking is post-hoc removal: someone takes your watermarked audio, compresses it, adds noise, or splices it, attempting to strip the mark. Both defenses are needed. Detection without watermarking means you cannot prove your own AI-generated calls are authorized. Watermarking without detection means you cannot catch clones that never passed through your system.

A third defense — cryptographic provenance signing (C2PA / Content Authenticity Initiative) — binds metadata about the audio's origin, model, and generation parameters to the file itself. This handles compliance: regulators and platform policies increasingly require AI-generated content to be identifiable. Provenance signing assumes a cooperative pipeline (you sign what you generate), while anti-spoofing assumes a hostile pipeline (you inspect what arrives). The three compose into a defense-in-depth stack, and a 2026 voice AI deployment that omits any layer is shipping a known vulnerability.

## The Concept

Automatic Speaker Verification (ASV) systems accept or reject a speaker claim by comparing an input utterance against an enrolled voiceprint. Spoofing attacks against ASV fall into four categories formalized by the ASVspoof challenge series: text-to-speech (TTS) synthesis, voice conversion (VC), replay of recorded bona fide speech, and deepfake neural synthesis. Anti-spoofing countermeasures (CMs) are binary classifiers that operate on audio features — either raw waveforms or handcrafted spectral representations like LFCC (Linear Frequency Cepstral Coefficients) and CQCC (Constant Q Cepstral Coefficients). The CM outputs a continuous score; a decision threshold separates bona fide speech from spoofed.

ASVspoof 2025 (the fifth challenge) evaluates CMs under an "in-the-wild" condition — attacks drawn from TTS and VC systems not present in training data, recorded in non-studio conditions, across roughly 2000 speakers versus the ~100 speakers in earlier editions. The challenge includes 32 attack algorithms spanning TTS, voice conversion, and adversarial perturbation. The primary evaluation metric is minimum tandem detection cost function (min-t-DCF), which jointly penalizes missed spoofing detections and false rejections of bona fide speech. State-of-the-art on ASVspoof 5 achieves approximately 7.23% Equal Error Rate (EER), compared to 0.42% EER on the older ASVspoof 2019 LA partition. That gap between lab and in-the-wild performance is the deployment reality: expect 5–10% EER on production audio.

[CITATION NEEDED — concept: ASVspoof 5 specific attack partition definitions and baseline results]

The two detection model families that dominate ASVspoof benchmarks are AASIST and RawNet2. **AASIST** (2021, updated through 2026) uses graph-attention layers operating on spectral features — it models the relationships between time-frequency bins as a graph and learns which subgraph patterns indicate synthesis artifacts. It is the current SOTA on the ASVspoof 5 countermeasure task. **RawNet2** operates directly on raw waveforms using a convolutional front-end followed by residual blocks and gated recurrent units. Both produce a single scalar score; both can be run on a single CPU in under 200ms for a 4-second clip. The choice between them in production comes down to feature engineering preference: AASIST's spectral graph captures artifacts in the frequency domain where TTS systems leave detectable traces, while RawNet2's end-to-end waveform processing avoids the assumption that the right features are known in advance.

Audio watermarking addresses a complementary problem. Rather than classifying audio as real or fake, watermarking embeds an imperceptible, recoverable signal into the waveform at generation time. **AudioSeal** (Meta, 2024) implements this with two neural networks trained adversarially. An **encoder** inserts a low-amplitude perturbation into the raw waveform — the perturbation is shaped to be below the perceptual threshold of human hearing but above the detection threshold of the companion network. A **detector** reads arbitrary audio segments and predicts per-sample watermark presence without needing the original clean signal. The adversarial training loop is what gives AudioSeal its robustness: the detector is trained on watermarked audio that has been passed through codec degradation (MP3, AAC), additive noise, resampling, and time-stretching, so the watermark survives these transformations at inference time.

The key architectural property of AudioSeal is **localization**. The detector outputs a per-sample binary prediction, not a single clip-level score. This means you can identify exactly which samples carry the watermark and which do not — enabling tampering detection. If you watermark a 10-second clip and someone replaces seconds 4–6 with different audio, the detector shows watermark presence on samples 0–3 and 7–10 but absence on samples 4–6. That spatial resolution is what distinguishes AudioSeal from earlier watermarking systems that produced only a single "watermark present/absent" verdict for the entire file.

```mermaid
flowchart TD
    A[Audio Source] --> B{Origin known?}
    B -->|Yes - Our pipeline| C[AudioSeal Encoder]
    B -->|No - Inbound/External| D[AASIST / RawNet2 CM]
    
    C --> E[Watermarked Audio]
    E --> F[AudioSeal Detector]
    F --> G{Per-sample watermark present?}
    G -->|All samples| H[Verified: Our content, unmodified]
    G -->|Partial samples| I[Tampered: Segments replaced]
    G -->|No samples| J[Unknown origin or stripped]
    
    D --> K[CM Score]
    K --> L{Score > threshold?}
    L -->|Yes| M[Bona fide speech]
    L -->|No| N[Spoofed / synthetic]
    
    H --> O[C2PA Sign + Ship]
    M --> O
    I --> P[Flag + Investigate]
    N --> P
    J --> Q[Run CM as fallback]
```

**WaveVerify** refers to a class of waveform-domain watermark verification approaches that attempt to improve on AudioSeal's localization and robustness by operating at multiple temporal resolutions simultaneously. The published architecture and benchmark results for a specific system called "WaveVerify" are not yet available in the literature I can access.

[CITATION NEEDED — concept: WaveVerify architecture, published results, and distinguishing mechanism vs. AudioSeal]

For practical purposes, AudioSeal and WavMark are the two open-source audio watermarking systems available today. WavMark uses a different embedding strategy — frequency-domain insertion via a trained encoder-decoder — and achieves higher bit capacity (up to 32 bits of payload vs. AudioSeal's 0-bit presence signal) at the cost of lower robustness to aggressive compression. The choice depends on whether you need to encode an identifier (use WavMark) or detect presence and tampering (use AudioSeal).

## Build It

### Beat 1: Anti-Spoofing Countermeasure Scoring

The first thing to build is a countermeasure pipeline that takes an audio clip, extracts features, and produces a spoofing score. We'll simulate the AASIST pipeline: load audio, extract a spectral representation, pass it through a classifier, and output a score relative to a threshold.

This code uses torchaudio for loading and transforms, and a lightweight CNN as a stand-in for the full AASIST graph-attention network. The model here is randomly initialized — in production you would load pretrained AASIST weights from the ASVspoof challenge repository. The pipeline structure, feature extraction, and scoring logic are identical to what the pretrained model uses.

```python
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)

sample_rate = 16000
duration_sec = 4.0
num_samples = int(sample_rate * duration_sec)

t = np.linspace(0, duration_sec, num_samples, endpoint=False)
bonafide_audio = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(num_samples)
spoofed_audio = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.random.randn(num_samples) * np.sin(2 * np.pi * 50 * t)

bonafide_tensor = torch.FloatTensor(bonafide_audio).unsqueeze(0)
spoofed_tensor = torch.FloatTensor(spoofed_audio).unsqueeze(0)

class SimpleCM(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, waveform):
        spec = self.mel_spec(waveform)
        spec_db = self.amplitude_to_db(spec)
        features = self.encoder(spec_db.unsqueeze(1) if spec_db.dim() == 3 else spec_db)
        score = self.classifier(features)
        return score.squeeze(-1)

cm_model = SimpleCM(sample_rate=sample_rate)
cm_model.eval()

with torch.no_grad():
    bonafide_score = cm_model(bonafide_tensor).item()
    spoofed_score = cm_model(spoofed_tensor).item()

threshold = 0.0

print(f"Bonafide clip score: {bonafide_score:.4f}")
print(f"Spoofed clip score:  {spoofed_score:.4f}")
print(f"Decision threshold:  {threshold:.4f}")
print(f"Bonafide classification: {'ACCEPT (bonafide)' if bonafide_score > threshold else 'REJECT (spoofed)'}")
print(f"Spoofed classification:  {'ACCEPT (bonafide)' if spoofed_score > threshold else 'REJECT (spoofed)'}")
```

Run this and you get two scores and two classifications. The scores from a randomly initialized model are not meaningful — but the pipeline is. In production, you swap `cm_model` for a pretrained AASIST checkpoint (available from the ASVspoof challenge GitHub), and the same code path produces real spoofing scores. The threshold value comes from the challenge's EER analysis on a held-out validation set.

The feature extraction step — Mel-spectrogram followed by amplitude-to-dB conversion — is where the detection signal lives. TTS systems produce spectrograms that differ from human speech in specific frequency bands: formant transitions are often too smooth, high-frequency noise floors are unnaturally flat, and phase coherence patterns differ from natural vocal tract resonances. AASIST's graph-attention layers learn to attend to these specific spectro-temporal regions. RawNet2 discovers similar patterns end-to-end from raw waveforms, but both approaches converge on detecting the same underlying artifacts: synthesis leaves traces.

### Beat 2: AudioSeal Watermark Embedding and Detection

Now we build the watermarking pipeline. AudioSeal's architecture consists of an encoder (embeds the watermark) and a detector (reads it back). The encoder takes a clean waveform and adds a learned perturbation. The detector takes any waveform and outputs per-sample predictions of watermark presence.

The following code implements a minimal version of this pipeline using PyTorch. The encoder and detector are small neural networks trained adversarially — the encoder learns to embed a perturbation that the detector can read but humans cannot hear, and the detector learns to read that perturbation even after degradation. This code runs the forward passes for embedding and detection; training the adversarial loop is covered in the exercises.

```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

sample_rate = 16000
duration_sec = 3.0
num_samples = int(sample_rate * duration_sec)

t = np.linspace(0, duration_sec, num_samples, endpoint=False)
clean_audio = 0.2 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.random.randn(num_samples)
clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)

class AudioSealEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=33, stride=1, padding=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=33, stride=1, padding=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=33, stride=1, padding=16),
            nn.Tanh(),
        )

    def forward(self, waveform, alpha=0.01):
        perturbation = self.encoder(waveform) * alpha
        watermarked = waveform + perturbation
        return watermarked, perturbation

class AudioSealDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=33, stride=4, padding=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=33, stride=4, padding=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=33, stride=4, padding=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=1, stride=1),
        )

    def forward(self, waveform):
        per_sample = self.detector(waveform)
        per_sample = torch.sigmoid(per_sample)
        return per_sample.squeeze(0).squeeze(0)

encoder = AudioSealEncoder()
detector = AudioSealDetector()
encoder.eval()
detector.eval()

with torch.no_grad():
    watermarked, perturbation = encoder(clean_tensor)
    clean_detected = detector(clean_tensor)
    watermarked_detected = detector(watermarked)

snr = 10 * torch.log10(
    torch.mean(clean_tensor ** 2) / torch.mean(perturbation ** 2 + 1e-10)
).item()

clean_mean = clean_detected.mean().item()
watermarked_mean = watermarked_detected.mean().item()

print(f"Clean audio samples: {num_samples}")
print(f"Perturbation SNR: {snr:.2f} dB (higher = more imperceptible)")
print(f"")
print(f"Detector on CLEAN audio:")
print(f"  Mean per-sample score: {clean_mean:.4f}")
print(f"  Max per-sample score:  {clean_detected.max().item():.4f}")
print(f"")
print(f"Detector on WATERMARKED audio:")
print(f"  Mean per-sample score: {watermarked_mean:.4f}")
print(f"  Max per-sample score:  {watermarked_detected.max().item():.4f}")
print(f"")
print(f"Watermark detection margin: {watermarked_mean - clean_mean:.4f}")
```

The perturbation SNR (Signal-to-Noise Ratio) tells you how imperceptible the watermark is. Production AudioSeal targets 30+ dB SNR — the perturbation is roughly 1/1000th the amplitude of the speech signal. The detection margin — the difference between the detector's mean score on watermarked vs. clean audio — tells you how separable watermarked audio is from unwatermarked audio. A margin above 0.3 on this metric is the practical threshold for reliable detection.

### Beat 3: Watermark Robustness Through Codec Degradation

The watermark must survive the transformations that happen to audio in the real world: MP3 compression, phone-line transmission, resampling, and noise addition. This code takes the watermarked audio from Beat 2, applies degradation, and checks whether the detector still reads the watermark.

```python
import torch
import numpy as np
import io
import wave
import struct

def degrade_mp3_simulation(audio_tensor, sample_rate, bitrate_kbps=64):
    audio_np = audio_tensor.squeeze(0).numpy()
    quantization_levels = bitrate_kbps * 8
    step = max(1, sample_rate // quantization_levels)
    degraded = audio_np.copy()
    for i in range(0, len(degraded), step):
        chunk = degraded[i:i+step]
        if len(chunk) > 0:
            mean_val = np.mean(chunk)
            degraded[i:i+step] = mean_val
    high_freq_noise = 0.005 * np.random.randn(len(degraded))
    degraded = degraded + high_freq_noise
    return torch.FloatTensor(degraded).unsqueeze(0)

def degrade_noise(audio_tensor, noise_level=0.05):
    noise = noise_level * torch.randn_like(audio_tensor)
    return audio_tensor + noise

def degrade_resample(audio_tensor, orig_sr=16000, target_sr=8000):
    audio_np = audio_tensor.squeeze(0).numpy()
    indices = np.arange(0, len(audio_np), orig_sr / target_sr).astype(int)
    downsampled = audio_np[indices]
    upsampled = np.interp(
        np.arange(len(audio_np)),
        np.arange(0, len(audio_np), orig_sr / target_sr)[:len(downsampled)],
        downsampled
    )
    return torch.FloatTensor(upsampled).unsqueeze(0)

with torch.no_grad():
    baseline = detector(watermarked).mean().item()
    
    degraded_mp3 = degrade_mp3_simulation(watermarked, sample_rate, bitrate_kbps=64)
    mp3_score = detector(degraded_mp3).mean().item()
    
    degraded_noise = degrade_noise(watermarked, noise_level=0.05)
    noise_score = detector(degraded_noise).mean().item()
    
    degraded_resample = degrade_resample(watermarked, orig_sr=16000, target_sr=8000)
    resample_score = detector(degraded_resample).mean().item()
    
    clean_baseline = detector(clean_tensor).mean().item()

print(f"{'Condition':<30} {'Detection Score':>16} {'Survives?':>12}")
print(f"{'-'*30} {'-'*16} {'-'*12}")
print(f"{'Clean (no watermark)':<30} {clean_baseline:>16.4f} {'---':>12}")
print(f"{'Watermarked (baseline)':<30} {baseline:>16.4f} {'YES':>12}")
print(f"{'MP3 simulation (64kbps)':<30} {mp3_score:>16.4f} {'YES' if mp3_score > clean_baseline + 0.05 else 'NO':>12}")
print(f"{'Additive noise (5% STD)':<30} {noise_score:>16.4f} {'YES' if noise_score > clean_baseline + 0.05 else 'NO':>12}")
print(f"{'Resample 16k->8k->16k':<30} {resample_score:>16.4f} {'YES' if resample_score > clean_baseline + 0.05 else 'NO':>12}")
print(f"")
print(f"Detection threshold (above clean baseline + 0.05): {clean_baseline + 0.05:.4f}")
```

With the randomly initialized detector, survival is not guaranteed — the adversarial training that produces robustness has not happened. In a trained AudioSeal model, the watermark survives MP3 compression down to 32kbps, additive noise up to 10% standard deviation, and resampling to 8kHz. The robustness comes directly from the adversarial training loop: during training, the detector sees degraded watermarked audio and is penalized for missing the watermark, so the encoder learns perturbations that persist through those specific transformations.

### Beat 4: Tampering Localization

AudioSeal's per-sample detection enables tampering localization — identifying exactly which samples of a clip were modified after watermarking. This code watermarks a clip, replaces a segment with unwatermarked audio, and runs the detector to visualize where the watermark is absent.

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

sample_rate = 16000
duration_sec = 5.0
num_samples = int(sample_rate * duration_sec)

t = np.linspace(0, duration_sec, num_samples, endpoint=False)
clean_audio = 0.2 * np.sin(2 * np.pi * 180 * t) + 0.05 * np.random.randn(num_samples)
clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)

with torch.no_grad():
    watermarked, _ = encoder(clean_tensor)

tamper_start = int(2.0 * sample_rate)
tamper_end = int(2.5 * sample_rate)
replacement = 0.2 * np.sin(2 * np.pi * 300 * t[tamper_start:tamper_end]) + \
              0.05 * np.random.randn(tamper_end - tamper_start)

tampered_audio = watermarked.clone()
tampered_audio[0, tamper_start:tamper_end] = torch.FloatTensor(replacement)

with torch.no_grad():
    detection_map = detector(tampered_audio)

window_size = sample_rate // 4
num_windows = len(detection_map) // window_size
window_scores = []

for i in range(num_windows):
    start = i * window_size
    end = start + window_size
    score = detection_map[start:end].mean().item()
    window_scores.append(score)

overall_score = detection_map.mean().item()
tampered_region_score = detection_map[tamper_start:tamper_end].mean().item()
intact_region_score = torch.cat([
    detection_map[:tamper_start],
    detection_map[tamper_end:]
]).mean().item()

print(f"Clip duration: {duration_sec:.1f}s ({num_samples} samples)")
print(f"Tampered region: {tamper_start/sample_rate:.1f}s - {tamper_end/sample_rate:.1f}s")
print(f"")
print(f"Per-0.25s-window detection scores:")
print(f"{'Time Range':>12}  {'Score':>8}  {'Status':>10}")
print(f"{'-'*12}  {'-'*8}  {'-'*10}")
for i, score in enumerate(window_scores):
    start_time = i * 0.25
    end_time = (i + 1) * 0.25
    status = "INTACT" if score > 0.05 else "TAMPERED"
    marker = " <-- REPLACED" if start_time >= 2.0 and end_time <= 2.5 else ""
    print(f"{start_time:>5.2f}-{end_time:<5.2f}s  {score:>8.4f}  {status:>10}{marker}")

print(f"")
print(f"Overall detection score:     {overall_score:.4f}")
print(f"Intact region avg score:     {intact_region_score:.4f}")
print(f"Tampered region avg score:   {tampered_region_score:.4f}")
print(f"Tampering confidence margin: {intact_region_score - tampered_region_score:.4f}")
```

The per-window detection scores reveal the tampering location. In production, this is how you detect if someone took your watermarked AI-generated call recording and spliced in a different segment — the spliced segment has no watermark, and the detector shows a gap. For a GTM team running voice AI outbound, this means you can prove that a problematic call recording was edited after it left your system: "seconds 0–14 and 18–22 carry our watermark; seconds 15–17 do not. Those two seconds were not generated by our pipeline."

## Use It

The Signal Machine pattern — building a system that captures, classifies, and routes signals from inbound and outbound interactions — depends on trust in the signal source. When a voice AI makes outbound calls, the call recording is a signal that feeds back into the routing and qualification pipeline. Anti-spoofing detection is how the Signal Machine filters inbound signals: if a prospect calls back and the audio is classified as synthetic, that is either a competitor probing your system or a man-in-the-middle attack on your IVR. The countermeasure model runs on every inbound voice channel, and clips scoring below threshold are flagged for human review rather than routed to an automated sequence.

Watermarking is how the Signal Machine establishes outbound signal provenance. Every audio clip generated by your AI SDR — the call audio itself, the voicemail drops, the voice-modulated demo walkthroughs — passes through the AudioSeal encoder before it leaves your system. When that audio appears later (a prospect forwards a voicemail, a compliance team requests a call recording, a regulator asks for proof of AI-generated content disclosure), the detector verifies the watermark and the per-sample map confirms the clip was not edited. This is the mechanism behind C2PA-compliant AI content labeling: the watermark is the cryptographic-adjacent proof that the audio originated from your model, and the C2PA manifest binds that proof to metadata about which model, which version, and which campaign generated it.

The deployment pattern for Inbound-Led Outbound connects these defenses directly. When an inbound lead comes in via voice channel, the anti-spoofing CM scores the audio in real time. If the score indicates bona fide speech, the Signal Machine routes the lead to the appropriate sequence using the embedding-based classification from Zone 06. If the score indicates synthetic audio, the lead is held for verification — because a synthetic inbound voice call to your AI SDR system is either a red team test, a competitor mapping your call flows, or an attempted prompt injection via audio channel. The watermarking layer runs in the opposite direction: outbound audio is watermarked at generation time, and any downstream system that needs to verify "did this come from us" runs the detector.

The practical integration looks like this: your voice AI pipeline generates audio, the AudioSeal encoder adds the watermark in the same inference pass, the audio is transmitted, and at every downstream touchpoint (CRM call log, compliance archive, analytics pipeline) the detector verifies provenance. The countermeasure model runs as a middleware service — a gRPC endpoint that accepts audio chunks and returns spoofing scores — sitting in front of any system that accepts voice input. Latency budget: AASIST inference on a single CPU takes ~150ms for a 4-second clip. AudioSeal detection takes ~50ms for the same clip. Both fit within a real-time voice processing pipeline.

[CITATION NEEDED — concept: ASVspoof 5 production deployment latency benchmarks on commodity CPU hardware]

## Ship It

Shipping voice anti-spoofing and watermarking into production requires three integration points and one operational decision. The integration points: (1) the AudioSeal encoder embedded in the voice generation pipeline, running as a post-processing step after TTS synthesis and before audio transmission; (2) the AudioSeal detector deployed as a verification service callable by downstream systems; (3) the AASIST countermeasure deployed as a real-time classification service on inbound audio channels. The operational decision is the threshold: where you set the CM decision boundary determines your false acceptance rate (synthetic audio classified as real) and false rejection rate (real speech classified as synthetic).

The threshold is a business decision, not a technical one. ASVspoof 5's min-t-DCF metric formalizes this tradeoff: the cost of a missed spoofing detection (an attacker's clone passes through) versus the cost of a false rejection (a real prospect's call is blocked). For a voice AI outbound system targeting enterprise prospects, a false rejection means a lost conversation with a potential customer — the cost is the pipeline value of that conversation. A false acceptance means a synthetic audio clip enters your system undetected — the cost depends on what that clip can do. If inbound voice triggers automated workflows (booking, qualification, data entry), a false acceptance could mean an attacker injects synthetic commands into your pipeline. Set the threshold conservatively (favor rejection) when inbound voice triggers automated actions; set it permissively when human review follows every inbound call.

The watermarking layer has a simpler shipping decision: watermark everything or watermark selectively. Watermarking everything — every TTS output, every voicemail, every demo audio — is the correct default. The computational cost is negligible (encoder inference is a single forward pass), and selective watermarking creates gaps in provenance coverage. The one exception is real-time streaming audio (live AI conversation): AudioSeal's encoder processes fixed-length segments, and chunked encoding introduces boundary artifacts. For streaming, use a segment size of 200ms with 50ms overlap, watermark each segment independently, and concatenate. The detector reads the concatenated stream without knowing about segment boundaries.

The compliance dimension is where watermarking meets regulation. The EU AI Act requires disclosure of AI-generated content. C2PA provides the metadata framework. The AudioSeal watermark provides the technical enforcement: if a piece of audio is disputed, the detector output is evidence that the audio passed through your pipeline. Store the per-sample detection map alongside the call recording — not just the binary "watermark present" verdict — because the map is what proves the recording was not edited after generation. This matters for dispute resolution: if a prospect claims your AI SDR said something inappropriate, the watermark map on the archived recording shows whether the specific segment in question was generated by your system or spliced in after the fact.

[CITATION NEEDED — concept: EU AI Act Article 50 transparency obligations for AI-generated audio content and technical compliance mechanisms]

## Exercises

**Easy.** Generate two synthetic audio clips — one with a clean sine wave (simulating natural speech patterns) and one with added high-frequency artifacts