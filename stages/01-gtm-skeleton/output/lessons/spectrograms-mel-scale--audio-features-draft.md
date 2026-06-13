# Spectrograms, Mel Scale & Audio Features

---

## Beat 1: Hook — Why Your Ear Isn't Linear

Human hearing operates on a logarithmic frequency scale — we can tell apart 200 Hz from 400 Hz easily, but struggle to distinguish 10,000 Hz from 10,200 Hz. Any ML pipeline that feeds raw waveforms or linear spectrograms into a model is forcing it to learn what biology already encodes. The Mel scale bakes this perceptual warp in upfront, reducing the dimensionality your model must learn.

---

## Beat 2: Concept — Three Representations of the Same Signal

**Time-domain waveform**: amplitude over time. No frequency information visible. **Linear spectrogram** (via Short-Time Fourier Transform): frequency over time, but allocates equal resolution to perceptually irrelevant high frequencies. **Mel spectrogram**: applies a triangular filterbank that warps frequency bins onto the Mel scale, compressing high-frequency resolution where human hearing is insensitive and preserving detail where it matters. **MFCCs** (Mel-Frequency Cepstral Coefficients): a further compression of the Mel spectrogram via DCT, retaining the 13–20 coefficients that capture vocal tract shape — the basis of speech recognition since the 1980s.

---

## Beat 3: Mechanism — From Waveform to Mel Spectrogram

1. **Frame the signal**: slide a window (typically 25 ms) across the waveform with a hop (typically 10 ms). Apply a Hann window to reduce spectral leakage.
2. **FFT each frame**: convert time-domain frame to frequency-domain. Output: complex values → magnitude spectrum.
3. **Apply Mel filterbank**: triangular filters spaced evenly on the Mel scale (not Hz scale). Each filter sums energy in its band. Output: one value per filter per frame.
4. **Log compression**: take log of filterbank energies. Human loudness perception is roughly logarithmic.
5. **(Optional) DCT to get MFCCs**: decorrelates log Mel energies, keeps lowest 13 coefficients, discards the rest.

Mel scale conversion formula (Hz → Mel): `m = 2595 · log10(1 + f/700)`. This is the mechanism that maps linear Hz to perceptual Mel.

---

## Beat 4: Implement — Build a Mel Spectrogram from Scratch

Generate a 440 Hz sine wave (A4) plus a 5000 Hz overtone. Compute a linear spectrogram via STFT. Compute a Mel spectrogram via filterbank. Print shape comparison to observe dimensionality reduction. Visualize both side-by-side to see how the Mel scale compresses the upper frequencies. Code uses `numpy`, `scipy.signal`, and `librosa` — all terminal-runnable, output confirmed via printed shapes and saved PNG.

**Exercise hook (easy):** Modify the signal to include a third harmonic at 200 Hz and confirm it appears in the Mel spectrogram output.
**Exercise hook (medium):** Write the Hz-to-Mel conversion function from the formula without using librosa's built-in, and verify output matches librosa's implementation.
**Exercise hook (hard):** Build the triangular filterbank manually (construct the matrix), apply it to a magnitude spectrum, and compare your result to librosa's `melspectrogram` output.

---

## Beat 5: Use It — Audio Features in GTM Context

Audio feature extraction is the front door for any pipeline that processes human speech — sales calls, support tickets, voice agents. The specific GTM application is **conversation intelligence** [CITATION NEEDED — concept: conversation intelligence platforms using MFCCs/Mel features for speaker identification, sentiment scoring, or talk-turn detection]. If your organization runs outbound voice sequences or ingests Zoom recordings for coaching, the raw audio must pass through this exact feature extraction pipeline before any downstream model (speaker diarization, keyword spotting, tone classification) can run. The Mel spectrogram is the standard input representation for production voice ML.

---

## Beat 6: Ship It — Production Considerations for Audio Pipelines

Audio resampling inconsistencies (44.1 kHz vs 16 kHz) silently break Mel feature comparability. Window size and hop length must be pinned at training time and locked at inference — a 512-sample window at 16 kHz is not equivalent to 512 samples at 22 kHz. Log-energy flooring prevents `-inf` values from silent frames. Batch inference requires padding signals to uniform length before STFT. These are the operational details that cause a model to score 95% in a notebook and 60% in production.

**Exercise hook (easy):** Write a validation function that checks if two audio files have matching sample rates before feature extraction.
**Exercise hook (medium):** Build a preprocessing pipeline that resamples any input to 16 kHz, floors log energies at -80, and returns a fixed-shape Mel spectrogram tensor.
**Exercise hook (hard):** Profile the latency of your feature extraction pipeline on 1-second, 10-second, and 60-second clips; identify the bottleneck and propose an optimization.

---

## Learning Objectives

1. **Implement** a Mel spectrogram from a raw waveform by applying STFT, Mel filterbank, and log compression in sequence.
2. **Compare** linear spectrogram and Mel spectrogram outputs, quantifying the dimensionality reduction and frequency warping.
3. **Configure** Mel filterbank parameters (n_fft, hop_length, n_mels, sample_rate) and predict how parameter changes affect output shape and frequency resolution.
4. **Extract** MFCCs from an audio signal and identify which coefficients capture vocal tract characteristics versus fine spectral detail.
5. **Diagnose** common production failures in audio feature pipelines (sample rate mismatch, silent frame handling, padding artifacts) and implement mitigations.