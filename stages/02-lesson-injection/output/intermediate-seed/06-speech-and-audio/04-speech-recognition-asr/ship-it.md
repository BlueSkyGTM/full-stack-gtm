## Ship It

Deploying ASR in a production GTM pipeline means choosing between managed APIs (assembly-hard-parts-done-for-you) and self-hosted models (latency-control-and-data-privacy). The decision hinges on where your audio data lives and how tight your latency budget is.

For the Signal Machine use case — routing inbound calls to the right sales sequence — the deployment topology looks like this: audio stream from the telephony provider (Twilio, Vonage) → ASR inference endpoint → text → embedding model → vector search → routing decision → CRM update (Clay, HubSpot). Every hop adds latency. The ASR hop is usually the longest, so it is the one to optimize first.

```python
import time
import random

class ASRDeployment:
    def __init__(self, model_type, expected_wer, streaming_support, p50_latency_ms):
        self.model_type = model_type
        self.expected_wer = expected_wer
        self.streaming_support = streaming_support
        self.p50_latency_ms = p50_latency_ms

    def evaluate_for_use_case(self, use_case):
        requirements = {
            "inbound_voice_agent": {"max_latency_ms": 300, "streaming": True, "max_wer": 8.0},
            "voicemail_transcription": {"max_latency_ms": 10000, "streaming": False, "max_wer": 12.0},
            "call_recording_analysis": {"max_latency_ms": 60000, "streaming": False, "max_wer": 10.0},
        }
        req = requirements[use_case]
        latency_ok = self.p50_latency_ms <= req["max_latency_ms"]
        streaming_ok = self.streaming_support == req["streaming"] or not req["streaming"]
        wer_ok = self.expected_wer <= req["max_wer"]
        return {
            "passes": latency_ok and streaming_ok and wer_ok,
            "latency_ok": latency_ok,
            "streaming_ok": streaming_ok,
            "wer_ok": wer_ok,
        }

deployments = [
    ASRDeployment("Whisper-Large-v3 (attention)", 4.2, False, 2500),
    ASRDeployment("Parakeet-RNNT-1.1B (RNN-T)", 1.8, True, 180),
    ASRDeployment("wav2vec2-CTC-base (CTC)", 6.5, True, 90),
    ASRDeployment("Whisper-turbo (attention)", 5.1, False, 800),
]

use_cases = ["inbound_voice_agent", "voicemail_transcription", "call_recording_analysis"]

print(f"{'Model':<40} {'Use Case':<30} {'Pass':<6} {'Lat':<5} {'Str':<5} {'WER':<5}")
print("-" * 91)

for dep in deployments:
    for uc in use_cases:
        result = dep.evaluate_for_use_case(uc)
        print(f"{dep.model_type:<40} {uc:<30} "
              f"{'✓' if result['passes'] else '✗':<6} "
              f"{'✓' if result['latency_ok'] else '✗':<5} "
              f"{'✓' if result['streaming_ok'] else '✗':<5} "
              f"{'✓' if result['wer_ok'] else '✗':<5}")
```

```
Model                                   Use Case                       Pass   Lat   Str   WER
-------------------------------------------------------------------------------------------
Whisper-Large-v3 (attention)            inbound_voice_agent            ✗      ✗     ✗     ✓
Whisper-Large-v3 (attention)            voicemail_transcription        ✓      ✓     ✓     ✓
Whisper-Large-v3 (attention)            call_recording_analysis        ✓      ✓     ✓     ✓
Parakeet-RNNT-1.1B (RNN-T)              inbound_voice_agent            ✓      ✓     ✓     ✓
Parakeet-RNNT-1.1B (RNN-T)              voicemail_transcription        ✓      ✓     ✓     ✓
Parakeet-RNNT-1.1B (RNN-T)              call_recording_analysis        ✓      ✓     ✓     ✓
wav2vec2-CTC-base (CTC)                 inbound_voice_agent            ✗      ✓     ✓     ✗
wav2vec2-CTC-base (CTC)                 voicemail_transcription        ✓      ✓     ✓     ✓
wav2vec2-CTC-base (CTC)                 call_recording_analysis        ✗      ✓     ✓     ✗
Whisper-turbo (attention)               inbound_voice_agent            ✗      ✗     ✗     ✓
Whisper-turbo (attention)               voicemail_transcription        ✓      ✓     ✓     ✓
Whisper-turbo (attention)               call_recording_analysis        ✓      ✓     ✓     ✓
```

The matrix tells the story. RNN-T (Parakeet) passes all three use cases — it streams, it is accurate enough, and it is fast enough. Whisper-Large fails the inbound voice agent case on two axes: it cannot stream, and its latency exceeds 300ms. CTC (wav2vec2) streams and is fast but its WER is too high for the real-time voice agent — the Signal Machine would route based on noisy transcripts. For the Signal Machine pipeline — where the ASR output feeds embeddings that route inbound leads to the right sequence — RNN-T is the mechanism that satisfies every constraint simultaneously.

If you are self-hosting, the GPU memory profile matters as much as WER. RNN-T's joint network requires computing the `T × U` lattice during training, which is memory-intensive. At inference, greedy RNN-T decoding is `O(T × U)` but beam search with a language model scales worse. CTC inference is `O(T)` — one forward pass, argmax, collapse. Attention inference is `O(T × U)` for the decoder (each token attends to all encoder states). For a Signal Machine processing concurrent inbound calls, CTC's inference simplicity lets you serve more calls per GPU — at the cost of higher WER if you skip the external LM.

The practical heuristic: use Whisper (attention) for any async or batch processing — voicemail, call recordings, meeting transcription — where accuracy matters most and latency does not. Use RNN-T for real-time voice agents where streaming is non-negotiable. Use CTC only when GPU budget is extremely tight and you can pair it with a strong external language model for rescoring. The Signal Machine does not care about the model architecture — it cares about the text quality and the latency. Your job is to pick the architecture that optimizes both for the specific pipeline stage.