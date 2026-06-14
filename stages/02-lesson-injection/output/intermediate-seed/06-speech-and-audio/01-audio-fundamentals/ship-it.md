## Ship It

The pipeline you just built — sample, window, FFT, extract features — is what every conversation intelligence tool runs internally. A production call-analytics system processes audio through this exact sequence before any ML model touches it. The difference between a prototype and a deployable tool is packaging this logic into a CLI that a go-to-market engineer can run against a directory of call recordings without understanding the DSP underneath.

Here is a self-contained CLI that analyzes any `.wav` file, prints sample-rate diagnostics, detects aliasing risk, and reports the spectral centroid for VAD-style segmentation. It generates a test file if none is provided, so it runs unmodified.

```python
#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.io.wavfile import write, read
import tempfile, os, sys

def generate_test_wav(path, sr=16000, duration=2.0):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 600 * t)
    signal += 0.02 * np.random.randn(n)
    write(path, sr, (signal * 32767).astype(np.int16))
    return path

def analyze_wav(path):
    sr, data = read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / 32768.0

    fft_result = np.fft.rfft(data)
    magnitudes = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(data), 1/sr)

    peak_idx = np.argmax(magnitudes[1:]) + 1
    peak_freq = freqs[peak_idx]
    spectral_centroid = np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-10)
    nyquist = sr / 2

    print(f"  File:             {os.path.basename(path)}")
    print(f"  Sample rate:      {sr} Hz")
    print(f"  Nyquist limit:    {nyquist:.0f} Hz")
    print(f"  Duration:         {len(data)/sr:.2f} s")
    print(f"  Samples:          {len(data)}")
    print(f"  Dominant freq:    {peak_freq:.1f} Hz")
    print(f"  Spectral centroid:{spectral_centroid:.1f} Hz")
    print(f"  VAD estimate:     {'speech' if spectral_centroid < 800 else 'noise/silence'}")

    if peak_freq > nyquist * 0.95:
        print(f"  WARNING: dominant freq near Nyquist — possible aliasing")
    if sr < 16000:
        print(f"  WARNING: sample rate below 16 kHz — consonant intelligibility at risk")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze WAV audio for call-analytics pipelines")
    parser.add_argument("path", nargs="?", default=None, help="Path to .wav file")
    parser.add_argument("--generate", action="store_true", help="Generate test WAV and analyze it")
    args = parser.parse_args()

    if args.path:
        analyze_wav(args.path)
    else:
        test_path = os.path.join(tempfile.gettempdir(), "call_sample.wav")
        generate_test_wav(test_path)
        print(f"No file provided. Generated test file: {test_path}\n")
        analyze_wav(test_path)
```

Save as `analyze_audio.py` and run `python analyze_audio.py` to see it generate a test file and analyze it. Run `python analyze_audio.py some_call.wav` to analyze a real recording. [CITATION NEEDED — concept: conversation intelligence / call analytics cluster in gtm-topic-map.md]. This is foundational for Zone III.