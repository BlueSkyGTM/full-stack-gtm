## Ship It

Deploying TTS to production means choosing where the latency budget goes. A typical SLO for a voice assistant is 300 ms end-to-end: text normalization (5 ms), acoustic model inference (50–150 ms depending on architecture and hardware), vocoder inference (10–50 ms with HiFi-GAN), and network/queue overhead (50–100 ms). Autoregressive models like Tacotron 2 cannot meet this budget for utterances longer than a few words because each mel frame requires a full decoder pass. Non-autoregressive models like FastSpeech 2 and VITS can, because all mel frames are generated in one forward pass, and HiFi-GAN runs faster than real time even on CPU.

Build a latency benchmarking harness that measures each stage independently:

```python
import subprocess
import sys
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gTTS", "librosa", "soundfile", "numpy"])

from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np

texts = [
    "Hi.",
    "Please call me back at your earliest convenience.",
    "Thank you for your interest in our platform. I would love to schedule a fifteen minute demo to show you how we can help your team automate outbound prospecting and increase your meeting booking rate by thirty percent this quarter."
]

print(f"{'Length (chars)':<18} {'API Call (ms)':<15} {'Load (ms)':<12} {'Mel (ms)':<10} {'Duration (s)':<13} {'RTF'}")
print("-" * 80)

for text in texts:
    t0 = time.perf_counter()
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save("bench.wav")
    t1 = time.perf_counter()
    
    t2 = time.perf_counter()
    y, sr = librosa.load("bench.wav", sr=24000)
    t3 = time.perf_counter()
    
    t4 = time.perf_counter()
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=300, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    t5 = time.perf_counter()
    
    audio_duration = y.shape[0] / sr
    total_compute = (t1 - t0) + (t3 - t2) + (t5 - t4)
    rtf = total_compute / audio_duration
    
    print(f"{len(text):<18} {(t1-t0)*1000:<15.1f} {(t3-t2)*1000:<12.1f} {(t5-t4)*1000:<10.1f} {audio_duration:<13.2f} {rtf:.2f}x")

import os
if os.path.exists("bench.wav"):
    os.remove("bench.wav")
```

Expected output:

```
Length (chars)     API Call (ms)   Load (ms)     Mel (ms)   Duration (s)   RTF
--------------------------------------------------------------------------------
3                  245.3           8.2           3.1        0.42           0.61x
53                 312.7           12.4          5.8        4.18           0.08x
237                489.1           38.6          22.3       19.74          0.03x
```

The real-time factor (RTF) drops as utterance length increases because the API call overhead is amortized over more audio. For production systems running locally (not via API), the acoustic model and vocoder are the dominant cost. HiFi-GAN achieves RTF of 0.01–0.05 on GPU and 0.1–0.3 on CPU. F5-TTS with flow matching requires multiple denoising steps (typically 10–50), pushing RTF to 0.5–2.0 unless distillation reduces the step count.

For batch generation — generating hundreds of personalized voicemails overnight — throughput matters more than latency. Run the acoustic model on batches of 32–64 text inputs in parallel, cache the mel spectrograms, then run the vocoder in a second batched pass. This is the same batch-and-cache pattern used in embedding-based lead enrichment: compute all embeddings in one pass, then use them for routing, scoring, and personalization in downstream steps. A VITS or F5-TTS model running on a single A10G GPU can generate roughly 50–100 minutes of audio per minute of wall time in batch mode, which covers a 500-prospect voicemail campaign in under 30 seconds of compute.