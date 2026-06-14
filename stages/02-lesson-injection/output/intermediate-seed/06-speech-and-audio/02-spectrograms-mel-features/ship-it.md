## Ship It

When deploying this in a production pipeline, you rarely compute spectrograms dynamically in pure Python. You use optimized C/C++ backends or hardware acceleration. `librosa` is the standard for offline batch processing and data preparation, but for real-time inference (like live call analysis or keyword spotting), tools like `torchaudio` or ONNX Runtime are preferred.

Here is how you extract MFCCs in PyTorch for a batch of signals. MFCCs apply a Discrete Cosine Transform (DCT) to the log-Mel spectrogram, keeping the lowest coefficients to capture the overall shape of the vocal tract.

```python
import torch
import torchaudio

print(f"Torchaudio version: {torchaudio.__version__}")

sample_rate = 16000
duration = 2.0
t = torch.arange(0, duration, 1.0/sample_rate).unsqueeze(0)
waveform = 0.5 * torch.sin(2 * torch.pi * 440 * t)

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=13,
    melkwargs={
        "n_fft": 400,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 64,
    }
)

mfcc = mfcc_transform(waveform)
print(f"Input waveform shape: {waveform.shape}")
print(f"Output MFCCs shape: {mfcc.shape}")
```

This implementation runs entirely on GPU tensors if available, allowing you to process thousands of audio files in parallel during model training or bulk feature extraction.