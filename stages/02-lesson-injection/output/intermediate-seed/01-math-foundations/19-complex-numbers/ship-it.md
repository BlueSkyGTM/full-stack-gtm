## Ship It

**Easy** — Write a function that converts rectangular complex numbers to polar form and back. Verify round-trip fidelity with assert statements.

```python
import numpy as np

def to_polar(z):
    return abs(z), np.angle(z)

def from_polar(r, theta):
    return r * np.exp(1j * theta)

test_values = [3+4j, 1+1j, -2+0j, 0+5j, -1-1j, 0.5-0.5j, 2-3j]

for z in test_values:
    r, theta = to_polar(z)
    z_back = from_polar(r, theta)
    assert np.isclose(z, z_back), f"Round-trip failed for {z}"
    print(f"  {z} -> r={r:.4f}, theta={np.degrees(theta):.2f} deg -> {z_back:.6f}")

print("All round-trip conversions passed.")
```

**Medium** — Implement a 1D DFT using only complex multiplication (no FFT call). Validate output against `numpy.fft.fft` on a test signal of length 64.

```python
import numpy as np

def dft_slow(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

N = 64
t = np.linspace(0, 1, N, endpoint=False)
x = np.sin(2 * np.pi * 3 * t) + 0.3 * np.cos(2 * np.pi * 7 * t)

X_slow = dft_slow(x)
X_fast = np.fft.fft(x)

print(f"Max difference: {np.max(np.abs(X_slow - X_fast)):.2e}")
print(f"Match: {np.allclose(X_slow, X_fast)}")

mags = np.abs(X_slow[:N//2])
peak_bin = np.argmax(mags)
print(f"Dominant frequency bin: {peak_bin} (expected 3)")
print(f"Magnitude at bin 3: {mags[3]:.2f}")
print(f"Magnitude at bin 7: {mags[7]:.2f}")
```

**Hard** — Compute and print the spectrogram of a generated chirp signal (frequency increasing over time). Identify the single frequency bin with highest magnitude at each time step.

```python
import numpy as np

duration = 2.0
sample_rate = 256
n_samples = int(duration * sample_rate)
t = np.linspace(0, duration, n_samples, endpoint=False)

f_start = 5
f_end = 50
instantaneous_freq = f_start + (f_end - f_start) * t / duration
phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate
chirp = np.sin(phase)

window_size = 64
hop_size = 32
n_windows = (n_samples - window_size) // hop_size

print(f"Chirp: {f_start} Hz to {f_end} Hz over {duration}s")
print(f"Windows: {n_windows}, window_size: {window_size}, hop: {hop_size}")
print(f"\n{'Window':>6} {'Time(s)':>8} {'Peak Bin':>8} {'Peak Freq':>9} {'|X|':>8}")
print("-" * 45)

dominant_bins = []
for w in range(n_windows):
    start = w * hop_size
    window = chirp[start:start + window_size]
    X = np.fft.fft(window)
    mags = np.abs(X[:window_size // 2])
    peak_bin = np.argmax(mags)
    peak_freq = peak_bin * sample_rate / window_size
    time_center = (start + window_size / 2) / sample_rate
    expected_freq = f_start + (f_end - f_start) * time_center / duration
    dominant_bins.append((peak_bin, peak_freq, time_center, expected_freq))
    print(f"{w:6d} {time_center:8.3f} {peak_bin:8d} {peak_freq:8.1f}Hz {mags[peak_bin]:8.2f}")

print(f"\nDetected frequency sweep: {dominant_bins[0][1]:.1f} Hz -> {dominant_bins[-1][1]:.1f} Hz")
print(f"Expected sweep: {f_start} Hz -> {f_end} Hz")

errors = [abs(d[1] - d[3]) for d in dominant_bins]
print(f"Mean frequency error: {np.mean(errors):.2f} Hz")
print(f"Max frequency error:  {np.max(errors):.2f} Hz")
```