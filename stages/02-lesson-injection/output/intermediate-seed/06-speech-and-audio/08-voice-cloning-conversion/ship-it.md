## Ship It

Shipping voice cloning to production requires three gates that do not exist in a typical TTS pipeline: consent verification, watermarking, and quality monitoring.

**Consent verification.** California AB 2905 (effective 2025) makes it unlawful to distribute a clone of a person's voice without their explicit consent. The EU AI Act (enforceable August 2026) requires disclosure that audio is AI-generated. Your pipeline needs a consent registry: a database mapping `speaker_id` to a signed consent record with timestamp, scope, and revocation capability. Before synthesis, check the registry. If consent is missing or revoked, refuse.

**Watermarking.** You must embed an inaudible signal in generated audio that identifies it as AI-generated. The C2PA (Coalition for Content Provenance and Authenticity) standard defines a metadata-based approach, but metadata strips easily. Audio-native watermarks — like Adobe's AudioSeal or Google's SynthID — embed patterns in the spectrogram that survive compression and format conversion. Your pipeline must apply the watermark before returning audio to the caller.

Here is a consent gate and watermark stub:

```python
import time
import json
import hashlib

class ConsentGate:
    def __init__(self):
        self.records = {}

    def grant(self, speaker_id, scope="commercial", consented_by=None):
        self.records[speaker_id] = {
            "speaker_id": speaker_id,
            "scope": scope,
            "granted_at": time.time(),
            "consented_by": consented_by or speaker_id,
            "active": True,
        }
        return self.records[speaker_id]

    def revoke(self, speaker_id):
        if speaker_id in self.records:
            self.records[speaker_id]["active"] = False
            return True
        return False

    def check(self, speaker_id):
        record = self.records.get(speaker_id)
        if record is None or not record["active"]:
            return False, "No active consent record"
        return True, record

def apply_watermark(audio_array, speaker_id, model_id="speecht5"):
    ts = int(time.time() * 1000)
    manifest = {
        "speaker_id": speaker_id,
        "model_id": model_id,
        "generated_at": ts,
        "hash": hashlib.sha256(
            f"{speaker_id}{model_id}{ts}".encode()
        ).hexdigest()[:16],
    }
    marker_freq = 18000
    sample_rate = 16000
    duration = len(audio_array) / sample_rate
    t = np.linspace(0, duration, len(audio_array))
    watermark_signal = 0.001 * np.sin(2 * np.pi * marker_freq * t)
    watermarked = audio_array + watermark_signal.astype(audio_array.dtype)
    return watermarked, manifest

gate = ConsentGate()
gate.grant("ae_001", scope="commercial_voicemail", consented_by="Jordan Lee")
gate.grant("ae_002", scope="commercial_voicemail", consented_by="Sam Rivera")

synthesis_requests = [
    {"speaker_id": "ae_001", "text": "Hi, this is Jordan..."},
    {"speaker_id": "ae_002", "text": "Hi, this is Sam..."},
    {"speaker_id": "ae_003", "text": "Hi, this is Casey..."},
]

for req in synthesis_requests:
    approved, info = gate.check(req["speaker_id"])
    if not approved:
        print(f"REFUSED: {req['speaker_id']} — {info}")
        continue

    fake_audio = np.random.randn(16000).astype(np.float32) * 0.1
    watermarked_audio, manifest = apply_watermark(
        fake_audio, req["speaker_id"]
    )
    print(f"APPROVED: {req['speaker_id']}")
    print(f"  Consent scope: {info['scope']}")
    print(f"  Watermark hash: {manifest['hash']}")
    print(f"  Marker at {18000}Hz, amplitude delta: "
          f"{np.abs(watermarked_audio - fake_audio).max():.6f}")
    print()
```

The third gate is **quality monitoring**. In production, run periodic speaker similarity checks: sample 1% of generated audio, run it back through the speaker encoder, and compare the extracted embedding against the original reference embedding. If cosine similarity drops below a threshold (typically 0.75), flag the model for retraining or the reference audio for re-recording. This is the same drift-detection pattern you would apply to any embedding-based system in a GTM stack — a Signal Machine's routing accuracy degrades when lead embeddings drift from their training distribution, and a voice clone's fidelity degrades when the model produces audio that drifts from the reference speaker's embedding.

For latency budgeting in a production voicemail pipeline, measure each stage: speaker embedding lookup (<5ms with an in-memory index), TTS inference (200-800ms with XTTS v2 on a single A10G, faster with Voicebox), vocoder synthesis (50-100ms with HiFi-GAN), and watermark application (<10ms). Total should land under 1 second for a 15-second clip. If you are using an API like ElevenLabs, add 200-500ms network latency. [CITATION NEEDED — concept: specific latency benchmarks for ElevenLabs API vs self-hosted XTTS v2]