# Speaker Recognition & Verification

## Learning Objectives

1. Extract speaker embeddings from audio and compute pairwise similarity scores
2. Implement a 1:1 speaker verification pipeline with configurable threshold
3. Compare 1:N speaker identification vs. 1:1 verification architectures and their error tradeoffs
4. Evaluate verification accuracy using equal error rate (EER) and DET curves
5. Integrate anti-spoofing checks to flag replay and synthesis attacks

---

## Beat 1: Hook — The Problem with "Who Is Speaking?"

Your sales team records 10,000 calls per week. Two questions: "Is this the same rep who called last time?" (verification) and "Which rep is this?" (identification). Humans answer both in seconds. Production systems need to answer in milliseconds at scale. Speaker recognition is the embedding problem applied to voice — extract a fixed-length vector from variable-length audio, then compare vectors. The mechanism is the same as face recognition, but the signal is noisier and spoofing is easier.

---

## Beat 2: Concept — Mechanisms

**Embedding extraction pipeline.** Raw audio → mel spectrogram → encoder → embedding vector. The encoder learns to map same-speaker utterances to nearby points in embedding space while pushing different-speaker utterances apart. Training uses triplet loss or angular margin softmax.

**Verification (1:1).** Given a claimed identity and an enrollment embedding, extract embedding from test audio, compute cosine similarity to enrollment, accept if score > threshold. Threshold selection directly trades false acceptance rate (FAR) against false rejection rate (FRR). Equal error rate (EER) is where FAR = FRR — the standard single-metric benchmark.

**Identification (1:N).** Extract embedding from unknown audio, compare against all enrolled speaker embeddings, return highest similarity. Scales linearly with enrollment size. Open-set identification adds a "none of the above" rejection threshold.

**Architectures.** D-vectors (LSTM-based, now outdated). X-vectors (TDNN-based, data augmentation, the standard baseline). ECAPA-TDNN (channel attention, current state-of-the-art on VoxCeleb). All produce 256–512 dimensional embeddings.

**Anti-spoofing.** Replay detection, speech synthesis detection, voice conversion detection. Separate binary classifier trained on bonafide vs. spoofed audio (ASVspoof challenge datasets).

---

## Beat 3: Demo — Working Code

Build a speaker verification system using `speechbrain` (pretrained ECAPA-TDNN on VoxCeleb). Extract embeddings from two audio files, compute cosine similarity, apply threshold, print accept/reject. Observable output: similarity score and decision.

Then extend to 1:N identification against a small enrolled speaker database (3 speakers, synthetic or sample audio). Print ranked similarity scores and predicted identity.

Anti-spoofing extension: run `speechbrain`'s pretrained ASVspoof model on a test utterance, print bonafide/spoof score.

*Exercise hooks:*
- Easy: Compute cosine similarity between two embeddings and print the score.
- Medium: Build enrollment + verification loop that accepts/rejects test audio against a stored embedding.
- Hard: Sweep threshold from 0.0 to 1.0, compute FAR and FRR at each point, print EER.

---

## Beat 4: Use It — GTM Redirect

Speaker recognition maps to **conversation intelligence and call analytics** (Zone 3 – Engagement). Specific application: speaker diarization in recorded sales calls — "who spoke when" — requires speaker embeddings to cluster segments by speaker identity. This is the mechanism behind tools like Gong and Chorus that attribute dialogue turns to reps vs. prospects.

Verification maps to **voice-based account verification** in high-touch sales or support workflows: confirm the caller matches a known contact before proceeding.

*Exercise hook:*
- Medium: Given a diarized transcript with speaker labels (A, B), write a function that matches each label to an enrolled speaker embedding and outputs "Speaker A → [Rep Name]" or "Speaker A → [Unknown]."

---

## Beat 5: Ship It — Production Considerations

**Latency.** Embedding extraction on 3–5 seconds of audio runs in <100ms on CPU with ONNX export. Verification is a single cosine similarity — negligible cost. Identification scales linearly with enrollment; use approximate nearest neighbor (FAISS) for >10K speakers.

**Audio quality.** Models trained on VoxCeleb (YouTube/interview audio) degrade on phone calls (8kHz, compression). Downsample training data to 8kHz or use models trained on telephone speech. Microphone variation is a larger source of error than model architecture.

**Threshold calibration.** EER threshold from benchmarks does not transfer to your domain. Collect labeled in-domain pairs, sweep threshold, calibrate to your FAR/FRR tolerance. A sales call use case tolerates higher FAR than a banking use case.

**Enrollment.** First-time enrollment needs ≥5 seconds of clean speech. Average multiple utterances for a stable centroid embedding. Re-enroll periodically — voice drifts over months.

**Security.** Without anti-spoofing, speaker verification is trivially bypassed with a replayed recording or a clone from elevenlabs. Deploy ASVspoof model as a gate before verification.

*Exercise hook:*
- Hard: Export a SpeechBrain encoder to ONNX, run inference using `onnxruntime`, benchmark latency over 100 iterations, print p50/p95/p99.

---

## Beat 6: Evaluate — Assessment Hooks

**Concept questions:**
- Explain why cosine similarity is used instead of Euclidean distance for speaker verification (hint: embedding normalization behavior).
- Given a system with EER = 2%, if you lower the threshold below the EER point, which error rate increases and which decreases?

**Implementation questions:**
- Write a function that takes two embedding numpy arrays and returns a verification decision given a threshold.
- Write a function that takes one embedding and a dictionary of {speaker_name: embedding} and returns the identified speaker and confidence score.

**Evaluation questions:**
- Given a list of (score, ground_truth) pairs, compute FAR, FRR, and EER.
- A speaker verification system has 500 enrolled users. You deploy it in open-set identification mode. Estimate the number of similarity computations per query and the latency at 0.01ms per cosine similarity.

**Anti-spoofing questions:**
- Explain why a replay attack defeats a naive speaker verification system and describe one detection mechanism.

---

## GTM Redirect Rules

- **Use It:** Conversation intelligence / call diarization (Zone 3 – Engagement). Speaker embeddings enable "who spoke when" in recorded GTM calls — the mechanism behind Gong/Chorus speaker attribution.
- **Ship It:** Voice-based caller verification for account-aligned sales workflows. Threshold calibration is domain-specific: sales tolerates different error rates than security.
- If the AI concept does not cleanly map to GTM: redirect is "foundational for Zone XX signal extraction pipelines" — speaker embeddings are a building block, not a GTM motion themselves.