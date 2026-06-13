# Real-Time Audio Processing

## Hook

You're building a voice agent or conversation intelligence tool. The audio doesn't wait — it arrives as a continuous stream of samples, and if your pipeline can't process chunks faster than they arrive, you drop data or stall the conversation. This lesson covers the mechanisms that make real-time audio work.

## Learn It

Introduces the streaming audio model: sample rates, bit depth, chunking, and the critical constraint that processing time per chunk must remain below chunk duration. Explains buffering strategies (ring buffers, double buffering) and the Fast Fourier Transform as the mechanism for moving between time and frequency domains. Introduces Voice Activity Detection (VAD) as a gate for downstream ASR. Names `pyaudio` for I/O, `numpy` for FFT, and `webrtcv3` for VAD — but only after explaining what problem each solves.

**Exercise hooks:**
- **Easy:** Record a 3-second WAV file, read it back with `wave`, print sample rate / num frames / duration.
- **Medium:** Chunk an audio file into 30ms frames, compute RMS energy per frame, print frames exceeding a silence threshold.
- **Hard:** Implement a sliding-window FFT over a real audio signal, print the dominant frequency per window.

## See It

Walks through a complete, runnable script that reads a WAV file in chunks, applies FFT to each chunk, detects whether the chunk contains speech or silence using an energy threshold, and prints a real-time transcript of "SPEECH" / "SILENCE" labels with timestamps. This demonstrates the full pipeline in observable, terminal-only output.

**Exercise hooks:**
- **Easy:** Modify the silence threshold and observe how the speech/silence labels change.
- **Medium:** Replace energy-based VAD with `webrtcv3` and compare the outputs side by side.

## Use It

Real-time audio processing is the mechanism behind conversation intelligence platforms that sales teams use for live coaching, objection handling, and talk-to-listen ratio analysis. The audio pipeline you just built — chunk, detect speech, extract features — is the same pipeline that feeds talk-time metrics and live sentiment to a rep's dashboard during a Zoom call. [CITATION NEEDED — concept: conversation intelligence architectures and their reliance on streaming audio chunking + VAD]

**Exercise hooks:**
- **Easy:** Compute talk-to-listen ratio from a two-channel call recording (one channel per speaker).
- **Medium:** Build a script that flags moments where one speaker monologues for more than 60 seconds without interruption.

## Ship It

Assembles the pieces into a production-style CLI tool that accepts a live microphone stream, applies VAD, writes speech chunks to disk, and prints real-time status. Handles graceful shutdown, resource cleanup, and parameterizes chunk size / sample rate / VAD aggressiveness via command-line arguments. No browser dependencies — terminal only.

**Exercise hooks:**
- **Easy:** Ship the tool with hardcoded defaults and confirm it records speech segments to files.
- **Medium:** Add a flag that computes and prints dominant frequency per speech chunk alongside the file output.
- **Hard:** Integrate `whisper` in streaming mode (segment-level) so each speech chunk is transcribed and printed before the next chunk arrives.

## Stretch It

Explores latencies below 10ms for truly interactive voice agents. Covers audio resampling, multi-channel beamforming, and the tradeoff between chunk size and transcription accuracy. References the WebRTC audio processing stack (AEC, NS, AGC) and when to use each. Points toward real-time voice AI agent architectures that close the loop: listen → decide → speak.

**Exercise hooks:**
- **Medium:** Benchmark processing time per chunk at 10ms, 30ms, and 100ms chunk sizes. Print a table showing whether each meets the real-time constraint.
- **Hard:** Build a minimal voice loop: record 3 seconds, transcribe with Whisper, print the transcription, measure end-to-end latency. Report whether it would support a natural conversation cadence.

---

## Learning Objectives

1. Build a chunked audio processing pipeline that reads, transforms, and classifies audio frames in real time.
2. Implement FFT-based frequency analysis over sliding windows and extract the dominant frequency per chunk.
3. Configure voice activity detection to separate speech from silence in a streaming audio stream.
4. Compare energy-based and WebRTC-based VAD approaches on the same audio input.
5. Evaluate whether a given chunk size and processing pipeline satisfies the real-time constraint for a target latency budget.

## GTM Redirect

This lesson's audio pipeline maps to **Zone 3 (Conversion) — Conversation Intelligence**: the chunk → VAD → feature extraction → ASR path is the same mechanism that powers live call coaching, talk ratio analysis, and real-time objection detection in sales tools. Where the connection to GTM is indirect (e.g., FFT fundamentals), the redirect is "foundational for Zone 3 conversation intelligence systems."