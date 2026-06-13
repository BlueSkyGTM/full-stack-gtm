# The Fourier Transform

## Learning Objectives
- Implement the FFT on a composite signal and extract constituent frequencies from the magnitude spectrum
- Explain the relationship between time-domain sample count and frequency-domain bin resolution
- Detect dominant periodic cycles in time-series data using spectral analysis
- Compare FFT-based cycle detection against moving-average methods and identify when FFT is the wrong tool

---

## Hook

You stare at a 90-day outbound reply-rate graph and see noise. The Fourier Transform sees three overlapping cycles you can't eyeball: a 7-day weekly rhythm, a 14-day sprint cadence, and a 30-day monthly pulse. This lesson teaches you to make those cycles visible and measurable.

---

## Mechanism

Start with a single sinusoid defined by amplitude, frequency, and phase. Stack several sinusoids to form a composite signal. The Discrete Fourier Transform (DFT) inverts this construction: given N time-domain samples, it produces N frequency bins, each holding a complex number encoding magnitude and phase at that frequency. The Fast Fourier Transform (FFT) exploits the symmetry of the twiddle factors $e^{-j2\pi kn/N}$ to reduce computation from $O(N^2)$ to $O(N \log N)$. The output is a complex array where `|X[k]|` gives the energy at frequency bin `k`, and the bin resolution equals sample rate divided by N.

---

## Code

Build a composite signal from three sine waves (5 Hz, 23 Hz, 50 Hz), compute the FFT with `numpy.fft.fft`, convert complex output to a magnitude spectrum, and print the top three dominant frequencies with their magnitudes. Code runs in terminal, prints detected frequencies to confirm correctness.

```python
import numpy as np

sr = 1000
t = np.arange(0, 1, 1/sr)
signal = 3.0 * np.sin(2 * np.pi * 5 * t) + 1.5 * np.sin(2 * np.pi * 23 * t) + 0.7 * np.sin(2 * np.pi * 50 * t)

fft_result = np.fft.fft(signal)
magnitudes = np.abs(fft_result[:len(fft_result) // 2])
freqs = np.fft.fftfreq(len(signal), 1/sr)[:len(fft_result) // 2]

top_indices = np.argsort(magnitudes)[::-1][:3]
for i in top_indices:
    print(f"Detected frequency: {freqs[i]:.1f} Hz | Magnitude: {magnitudes[i]:.2f}")
```

Observable output:
```
Detected frequency: 5.0 Hz | Magnitude: 1500.00
Detected frequency: 23.0 Hz | Magnitude: 750.00
Detected frequency: 50.0 Hz | Magnitude: 350.00
```

---

## Use It

[CITATION NEEDED — concept: GTM cluster for time-series seasonality in pipeline/revenue metrics]. The FFT is the frequency-decomposition step in revenue cycle analysis. Feed it 180 days of daily outbound reply rates and it returns the dominant cycle periods buried in the noise: 7-day, 14-day, 30-day. This is the spectral analysis layer underneath any seasonality-aware forecasting model in Zone 2 (Pipeline Analytics). If your pipeline prediction model treats daily data as acyclic, it is leaving signal on the table.

---

## Ship It

Package the FFT-based cycle detector as a function that accepts a pandas Series with a datetime index, computes the FFT, and returns a ranked list of dominant cycle periods in human-readable units (days, weeks). Write the output to a JSON report: `{"dominant_cycles": [{"period_days": 7.1, "strength": 0.82}, {"period_days": 29.8, "strength": 0.41}]}`. Wire this into a weekly analytics cron job. GTM redirect: this is the automated seasonality detector for [CITATION NEEDED — concept: GTM recurring revenue cycle detection pipeline].

---

## Check It

- **Easy:** Given the printed magnitude spectrum from the code example, identify which three frequencies carry the most energy and verify they match the input signal.
- **Medium:** Modify the code to add Gaussian noise (`np.random.normal`) to the composite signal at increasing noise levels (SNR 10dB, 3dB, 0dB). At what noise level does the FFT fail to recover the 50 Hz component? Print the recovered frequencies at each level.
- **Hard:** Load a CSV of 365 days of synthetic daily pipeline creation data (provided), compute the FFT, extract the top three cycle periods, and compare against a 7/14/30-day moving-average decomposition. Print both methods' results side by side with execution time. State where FFT is the wrong tool and why.