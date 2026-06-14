# Real-Time Audio Processing

## Learning Objectives

- Compute per-chunk processing latency and verify it stays below chunk duration to prevent data loss.
- Implement an energy-based Voice Activity Detection (VAD) gate that classifies 30 ms audio frames as speech or silence.
- Apply the Fast Fourier Transform to extract dominant frequencies from streaming audio chunks.
- Build a command-line tool that streams audio from a file at real-time speed, applies VAD, and writes detected speech segments to disk.
- Calculate talk-to-listen ratio and detect monologue segments from multi-channel call recordings.

## The Problem

Audio does not wait for you. When a person speaks into a microphone, their voice arrives as a continuous stream of discrete samples — 16,000 of them per second at telephone quality, 48,000 at music quality. Your code receives these samples in fixed-size blocks called chunks or frames. A 30 ms frame at 16 kHz contains exactly 480 samples. If your processing of that frame takes longer than 30 ms, the next frame is already arriving while you are still working. You either drop it or stall the pipeline. There is no third option.

This constraint drives every architectural decision in real-time audio. Batch processing lets you load an entire file, process it at your leisure, and write the result. Real-time processing imposes a hard deadline on every frame. The deadline is the frame duration itself. A voice assistant that takes two seconds to respond to a user feels broken. Human conversational turn-taking latency is approximately 230 ms — the silence-to-response window that feels natural. Anything above 500 ms feels sluggish; above 1500 ms feels like a satellite call.

The budget for a full hear → understand → respond → speak loop in modern systems looks roughly like this:

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) achieved 200 ms full-duplex by collapsing these stages into a single model. GPT-4o-realtime (2024) clocks approximately 320 ms end-to-end. Cascaded pipelines built in 2022 shipped at 2500 ms. The 10× improvement came from three techniques: streaming everywhere (no file-level buffering), asynchronous pipelining with partial results (start TTS before ASR finishes