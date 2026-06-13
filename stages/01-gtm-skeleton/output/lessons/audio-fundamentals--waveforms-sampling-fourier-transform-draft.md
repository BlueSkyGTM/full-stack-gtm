# Audio Fundamentals — Waveforms, Sampling, Fourier Transform

---

## Hook

You've recorded a sales call. It's a `.wav` file — binary data you can't read. Three concepts stand between you and programmatic control of that audio: waveforms (what sound *is*), sampling (how it becomes data), and the Fourier Transform (how you decompose it into frequencies). This lesson builds that foundation.

---

## Learn It

**Waveforms.** Sound is pressure variation over time. A pure tone is a sine wave defined by three parameters: frequency (cycles per second, Hz), amplitude (loudness), and phase (offset). Complex sounds are sums of sine waves — this is Fourier's theorem.

**Sampling.** Continuous pressure waves become discrete arrays via sampling at a fixed rate. The Nyquist-Shannon theorem constrains this: you must sample at ≥2× the highest frequency present, or you get aliasing (false low frequencies). CD audio uses 44.1 kHz because human hearing tops out near 20 kHz. Each sample is quantized to a bit depth (16-bit = 65,536 levels).

**Fourier Transform.** The DFT converts a time-domain signal into frequency-domain representation: a list of complex numbers whose magnitudes reveal how much energy exists at each frequency bin. The FFT (Cooley-Tukey, 1965) computes the DFT in O(n log n) instead of O(n²), making real-time spectral analysis viable.

---

## See It

Generate a 440 Hz sine wave at 44,100 Hz sample rate. Print the first 10 samples to confirm waveform shape. Compute the FFT with `numpy.fft.rfft` and print the index of the peak magnitude — it should map to ~440 Hz. Then deliberately undersample the same signal at 500 Hz (below Nyquist) and print the peak frequency to show aliasing in action. All output to terminal, no plots needed.

---

## Use It

**GTM Redirect:** This mechanism is foundational for Zone III [CITATION NEEDED — concept: conversation intelligence / call analytics cluster in gtm-topic-map.md]. Audio feature extraction is the input layer for any pipeline that transcribes, scores, or classifies sales calls.

**Exercise hook (easy):** Generate a composite signal (200 Hz + 600 Hz) and print the two dominant frequencies from FFT output.

**Exercise hook (medium):** Load a real `.wav` file using `scipy.io.wavfile.read`, compute the spectral centroid (weighted mean frequency), and print it — this metric distinguishes voiced speech from silence/noise and is used in voice-activity detection for call recording segmentation.

**Exercise hook (hard):** Implement a rolling window FFT (spectrogram) on a 5-second synthetic signal where frequency changes over time. Print the dominant frequency per window to demonstrate time-frequency resolution tradeoff controlled by window size.

---

## Ship It

**GTM Redirect:** Any conversation intelligence tool (Gong, Chorus, personal builds) processes audio through this exact pipeline: sample → window → FFT → features → downstream model. This is foundational for Zone III.

**Exercise hook (medium):** Write a CLI script that accepts a `.wav` file path, computes and prints: (1) duration in seconds, (2) sample rate, (3) top-5 frequency peaks by magnitude, (4) spectral centroid. This is the diagnostic tool you reach for when a transcription pipeline produces garbage and you need to check whether the input audio is degraded.

---

## Assess It

**Question pool (grounded in Learn It and See It):**

1. A signal contains frequency components up to 8,000 Hz. What is the minimum sample rate required to avoid aliasing? (Tests Nyquist theorem recall.)

2. You run `rfft` on 1024 samples at 16,000 Hz. What is the frequency resolution (Hz per bin) of the output? (Tests the relationship between window length, sample rate, and bin width.)

3. A 600 Hz sine wave is sampled at 800 Hz. What aliased frequency appears in the spectrum? (Tests aliasing calculation: fold the true frequency into the Nyquist range.)

4. Print the first 5 values of `np.fft.rfft([1, 0, -1, 0])`. Which bin has the largest magnitude, and what frequency does it represent at a 4 Hz sample rate? (Tests interpretation of FFT output on real input.)

---

**Learning Objectives (docs/en.md):**

1. Generate discrete waveforms from frequency, amplitude, and sample rate parameters.
2. Detect aliasing by comparing peak FFT frequency against the known input frequency.
3. Compute and interpret FFT output to identify dominant frequencies in a signal.
4. Calculate spectral centroid from magnitude spectrum as a single-number audio feature.
5. Evaluate the tradeoff between time and frequency resolution when selecting FFT window size.