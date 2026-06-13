# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify

## Hook

Voice cloning services now need under 3 seconds of sample audio to produce a convincing replica. GTM teams deploying voice AI—conversational outbound, AI SDR calls, voice-modulated demos—face a trust problem: how does the receiver know the voice is authorized, and how do you prove your generated audio came from your system and not a competitor's clone? This lesson covers the two complementary defenses: spoofing detection (is this voice real?) and watermarking (did this audio originate from our pipeline?).

## Concept

Automatic Speaker Verification (ASV) systems accept or reject a speaker claim. Spoofing attacks against ASV fall into four categories defined by the ASVspoof challenge series: text-to-speech (TTS), voice conversion (VC), replay, and deepfake synthesis. Anti-spoofing models—called countermeasures (CMs)—operate as binary classifiers on audio features. Separately, audio watermarking embeds an imperceptible, recoverable signal into generated audio so provenance can be verified later. These two mechanisms compose: watermarking proves origin, anti-spoofing detects unauthorized origin.

## Mechanism

### ASVspoof 5 Protocol

ASVspoof 2025 (the fifth challenge) evaluates CMs under the "in-the-wild" condition—attacks drawn from unknown TTS/VC systems not seen during training. The evaluation metric is minimum tandem detection cost function (min-t-DCF), which jointly penalizes missed spoofing detections and false rejections of bona fide speech. CMs receive raw waveforms or extracted features (e.g., LFCC, CQCC) and output a score; a threshold separates bona fide from spoof.

[CITATION NEEDED — concept: ASVspoof 5 specific attack partition definitions and baseline results]

### AudioSeal Watermarking

AudioSeal (Meta, 2024) embeds a localized, detectable watermark into audio at the sample level. The mechanism: a neural encoder inserts a low-amplitude perturbation into the waveform. A neural detector reads arbitrary audio segments and predicts per-sample watermark presence without needing the original clean signal. The detector is trained adversarially against the encoder to resist removal attacks (compression, noise, resampling). Key property: localization—the detector can identify which specific samples carry the watermark, enabling tampering detection.

[CITATION NEEDED — concept: AudioSeal adversarial training robustness benchmarks against codec degradation]

### WaveVerify

[CITATION NEEDED — concept: WaveVerify architecture, published results, and distinguishing mechanism vs. AudioSeal]

**Exercise hooks:**
- (Easy) Given two audio clips (one bona fide, one TTS-synthesized), run a pretrained CM and report the score for each.
- (Medium) Embed a watermark into a generated audio file using AudioSeal, then pass it through MP3 compression and verify the watermark survives.
- (Hard) Implement a tampering detector: watermark a clean clip, replace a 0.5s segment with unwatermarked audio, and use the per-sample detector output to locate the edit boundary.

## Implement

Load a pretrained anti-spoofing model (AASIST, trained on ASVspoof 2019/2021 data) and score two audio files. Separately, embed and detect a watermark using AudioSeal's released weights. All code runs in terminal with observable output.

```python
import torch
import torchaudio
import json

print(f"torchaudio version: {torchaudio.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

waveform_bona_fide, sr = torchaudio.load("bona_fide_sample.wav")
waveform_spoof, sr2 = torchaudio.load("spoof_sample.wav")

print(f"Sample rate: {sr}")
print(f"Bona fide duration: {waveform_bona_fide.shape[1] / sr:.2f}s")
print(f"Spoof duration: {waveform_spoof.shape[1] / sr2:.2f}s")

from audioseal import AudioSeal

detector = AudioSeal.load_from_pretrained("audioseal_detector_16bits")
encoder = AudioSeal.load_from_pretrained("audioseal_generator_16bits")

watermarked = encoder(waveform_bona_fide.unsqueeze(0), sample_rate=sr, alpha=0.5)

result = detector(watermarked, sample_rate=sr)
print(f"Watermark detected in watermarked audio: {result}")

result_clean = detector(waveform_bona_fide.unsqueeze(0), sample_rate=sr)
print(f"Watermark detected in original audio: {result_clean}")

torchaudio.save("watermarked_output.wav", watermarked.squeeze(0), sr)
print("Saved watermarked_output.wav")
```

**Exercise hooks:**
- (Easy) Modify `alpha` and observe how SNR changes between original and watermarked audio.
- (Medium) Apply `torchaudio.transforms.MuLawEncoding` to the watermarked audio, then detect. Does the watermark survive telephony compression?
- (Hard) Watermark two different clips with the same encoder, concatenate them, and use the detector's per-sample output to find the splice point.

## Use It

**GTM Redirect: Zone 2 — Enrichment & Identity Verification, Zone 3 — Engagement (Voice AI Outbound)**

When deploying AI-generated voice in outbound sequences (AI SDR calls, personalized video voiceovers), the receiving side—prospect's fraud detection, enterprise compliance—will increasingly run anti-spoofing CMs on inbound audio. Watermarking your generated audio serves two purposes: (1) you can prove to compliance teams that a call originated from your licensed system, not an unauthorized clone, and (2) you enable post-call forensic verification if a dispute arises. For enrichment, voice biometric verification of prospects (confirming a voicemail matches a known speaker) feeds signal back into the scoring waterfall.

[CITATION NEEDED — concept: enterprise voice authentication adoption rates in sales technology stacks]

**Exercise hooks:**
- (Easy) Write a compliance log entry that records the watermark payload, timestamp, and campaign ID for a generated outbound call.
- (Medium) Build a function that takes an audio file, runs anti-spoofing detection, and returns a risk score alongside the enrichment data for that prospect.
- (Hard) Simulate an adversarial scenario: clone a voice from 5 seconds of sample audio, watermark the clone, and test whether the anti-spoofing CM still flags it. Document the tension between watermark presence (proves origin) and spoof detection (flags synthetic).

## Ship It

Production deployment of voice anti-spoofing and watermarking requires three decisions: (1) where in the pipeline to place the CM—real-time during calls adds latency (>200ms for AASIST on CPU), so batch post-call scoring may be the initial ship; (2) watermark payload capacity—AudioSeal supports 16-bit messages, enough for a system ID + campaign hash but not a full UUID; (3) robustness threshold—higher `alpha` improves detection after compression but degrades perceived audio quality. Ship a batch pipeline first: generate audio → watermark → store provenance record → post-delivery CM scoring on any inbound audio your system receives.

[CITATION NEEDED — concept: AudioSeal 16-bit payload capacity specification and bit error rate under codec degradation]

**Exercise hooks:**
- (Easy) Write a shell script that processes a directory of `.wav` files through the detector and outputs a CSV of filename, watermark_present, spoof_score.
- (Medium) Implement a FastAPI endpoint that accepts audio uploads, runs both CM scoring and watermark detection, and returns a JSON verdict. Measure p95 latency.
- (Hard) Set up a nightly batch job that re-scores all outbound call recordings from the past 24 hours against an updated CM model. Write a migration script that handles model version swaps without downtime.