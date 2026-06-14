## Ship It

Deploying voice generation to production requires guardrails that don't exist in a single API call. The most critical is **voice cloning consent**. Extracting a speaker embedding from someone's voice and using it to generate speech they never spoke is regulated in several jurisdictions. The EU AI Act (Article 50) requires disclosure of AI-generated content in interactions with people. The FTC has taken enforcement action under Section 5 against deceptive uses of cloned voices. State-level biometric laws (Illinois BIPA, Texas CIRA) cover voiceprints in some interpretations. ElevenLabs requires explicit consent confirmation before cloning a voice through their API. If your pipeline uses a cloned voice — not a pretrained one — you need a consent record tied to that voice ID.

The second production concern is **quality validation at scale**. When generating hundreds or thousands of files, you cannot listen to each one. The spectral analysis from the Build It section becomes a CI step: every generated file is checked for silence (RMS energy below threshold), excessive noise (spectral centroid outside speech range), and duration constraints (too short means truncated text, too long means the model rambled). Files that fail are flagged for human review or regenerated.

The third concern is **cost monitoring**. TTS APIs charge per character or per minute of generated audio. A batch run of 5,000 personalized clips at 150 characters each is 750,000 characters — at OpenAI TTS pricing ($15/1M characters for the standard model), that's $11.25 per batch. ElevenLabs charges per character with tier-dependent limits. Neither is expensive in absolute terms, but unmonitored retries on failures can multiply costs. Logging character counts and API response times per file makes the cost auditable.

This script implements the quality validation layer. Run it over a directory of generated audio files before they enter your outbound sequence. Files that fail any check are excluded — better to send no audio than to send a prospect a silent file or, worse, a glitch that sounds like a malfunction.

```python
import os
import librosa
import numpy as np

audio_dir = "outbound_audio"

SILENCE_THRESHOLD = 0.005
CENTROID_MIN = 300
CENTROID_MAX = 4000
DURATION_MIN = 2.0
DURATION_MAX = 60.0

results = []
for filename in sorted(os.listdir(audio_dir)):
    if not filename.endswith('.mp3'):
        continue

    filepath = os.path.join(audio_dir, filename)
    y, sr = librosa.load(filepath, sr=22050)

    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    duration = len(y) / sr

    rms_mean = float(np.mean(rms))
    centroid_mean = float(np.mean(centroid))

    issues = []
    if rms_mean < SILENCE_THRESHOLD:
        issues.append("SILENCE")
    if centroid_mean < CENTROID_MIN or centroid_mean > CENTROID_MAX:
        issues.append("FREQUENCY_ANOMALY")
    if duration < DURATION_MIN:
        issues.append("TOO_SHORT")
    if duration > DURATION_MAX:
        issues.append("TOO_LONG")

    status = "PASS" if not issues else "FAIL:" + ",".join(issues)

    results.append({
        "filename": filename,
        "duration": duration,
        "rms": rms_mean,
        "centroid_hz": centroid_mean,
        "status": status,
    })

    print(f"[{status:<25}] {filename}")
    print(f"  Duration: {duration:.1f}s | RMS: {rms_mean:.6f} | Centroid: {centroid_mean:.1f}Hz")

passed = sum(1 for r in results if r["status"] == "PASS")
failed = len(results) - passed
print(f"\n=== Validation Report ===")
print(f"Total files: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
if failed > 0:
    print("Failed files:")
    for r in results:
        if r["status"] != "PASS":
            print(f"