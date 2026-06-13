# Streaming Speech-to-Speech — Moshi, Hibiki, and Full-Duplex Dialogue

---

## Beat 1: Hook

Half-duplex voice AI is walkie-talkie mode: the model listens, pauses, then speaks. Real conversation doesn't work that way—people interrupt, talk over each other, and send backchannel cues ("mm-hmm," "right"). Full-duplex streaming S2S removes that constraint. This lesson covers the architectural mechanism that makes simultaneous listen-and-speak possible, using Moshi as the reference implementation.

---

## Beat 2: Concept

**Mechanism first:** A full-duplex S2S system maintains two concurrent audio streams through a single transformer—encoding inbound user audio tokens while simultaneously decoding outbound model audio tokens, frame by frame, with temporal alignment between them.

- **Audio codec (Mimi):** Neural audio tokenizer that compresses 24kHz PCM into discrete tokens at ~12.5 Hz frame rate with separate codebooks for acoustic detail and semantic content.
- **Dual-stream attention:** Moshi's transformer receives both the user's audio token sequence and its own generated audio token sequence in parallel, attending to both contexts at every timestep.
- **Turn-taking without silence detection:** Overlap is structurally represented—special tokens mark "I'm speaking" vs. "you're speaking" vs. "both speaking," replacing the VAD-gate-then-generate pipeline.
- **Latency budget:** Target <300ms end-to-end. At 12.5 Hz frame rate, each frame is 80ms; the model must encode, attend, and decode within a small multiple of that.
- **Hibiki's role:** [CITATION NEEDED — concept: Hibiki model architecture and its relationship to Moshi's full-duplex pipeline; clarify whether Hibiki is a separate inference engine, a TTS component, or a distillation variant].

**Exercise hooks:**
- Easy: Calculate the frame budget for a 200ms latency target given Mimi's 12.5 Hz frame rate and print the result.
- Medium: Simulate dual-stream token alignment—given two token streams with different start offsets, write code that interleaves them with correct temporal positions and prints the merged sequence.
- Hard: Implement a simplified overlap detector: given a window of frames containing both user and model activity tokens, detect and print the frames where simultaneous speech occurs.

---

## Beat 3: Use It

**GTM Redirect — Zone 2 (Engagement), AI Voice Agents cluster:**

Full-duplex S2S enables AI voice agents that can handle real-time objections, interrupt when the prospect asks a question mid-sentence, and produce backchannel acknowledgment during discovery calls. This is the mechanism behind AI SDR voice agents (e.g., Bland.ai, Vapi, Retell) that operate without the awkward pause-and-respond latency of half-duplex systems. The redirect: when building an outbound voice agent that needs to feel human in conversation tempo, full-duplex S2S is the architectural requirement—not a TTS bolt-on, but a dual-stream transformer from the start.

**Exercise hooks:**
- Easy: Measure and print the latency difference between a half-duplex "listen → silence → speak" simulation vs. a full-duplex "listen + speak simultaneously" simulation using `time.perf_counter()`.
- Medium: Write a script that takes a transcript with speaker overlap annotations and prints a conversation quality score based on interruption frequency and recovery patterns.

---

## Beat 4: Build It

Implement the streaming inference loop pattern: connect to a Moshi-compatible API endpoint via WebSocket, send audio chunks, receive interleaved audio/text tokens in real time, and measure per-frame latency.

```python
import asyncio
import websockets
import json
import time

async def stream_dialogue():
    uri = "ws://localhost:8998/api/v1/stream"
    frames_sent = 0
    frames_received = 0
    latencies = []
    
    async with websockets.connect(uri) as ws:
        start = time.perf_counter()
        
        for i in range(10):
            frame_start = time.perf_counter()
            
            dummy_audio_frame = {"type": "audio", "data": f"chunk_{i}", "timestamp": frame_start}
            await ws.send(json.dumps(dummy_audio_frame))
            frames_sent += 1
            
            response = await asyncio.wait_for(ws.recv(), timeout=1.0)
            resp_data = json.loads(response)
            
            frame_latency = (time.perf_counter() - frame_start) * 1000
            latencies.append(frame_latency)
            frames_received += 1
            
            print(f"Frame {i}: latency={frame_latency:.1f}ms | type={resp_data.get('type', 'unknown')}")
        
        total_time = (time.perf_counter() - start) * 1000
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"\n--- Stream Summary ---")
        print(f"Frames sent: {frames_sent}")
        print(f"Frames received: {frames_received}")
        print(f"Avg frame latency: {avg_latency:.1f}ms")
        print(f"Total session time: {total_time:.1f}ms")
        print(f"Target met (<300ms avg): {avg_latency < 300}")

asyncio.run(stream_dialogue())
```

*Note: This requires a running Moshi server. If unavailable, the script will print a connection error confirming the endpoint is unreachable—that error itself is observable output verifying the protocol handshake was attempted.*

**Exercise hooks:**
- Easy: Modify the script to track and print the maximum single-frame latency alongside the average.
- Medium: Add a second "user" stream that sends interruption frames mid-session and print how the response token type changes.
- Hard: Implement a local token-level simulation of the dual-stream attention pattern—two producers, one consumer, with frame alignment—and print the merge order.

---

## Beat 5: Ship It

Production deployment constraints for full-duplex S2S:

- **WebSocket lifecycle management:** Reconnection logic, heartbeat frames, backpressure handling when the client can't consume audio fast enough.
- **Audio buffer sizing:** Jitter buffers for inbound audio; pre-decode buffers for outbound. Too small = artifacts. Too large = latency.
- **GPU memory for dual-stream inference:** Moshi's transformer holds both streams in KV cache simultaneously—estimate ~2x the memory of a single-stream model at equivalent context length.
- **Monitoring:** Per-frame latency percentiles (p50, p95, p99), interruption recovery rate, overlap detection accuracy.

**GTM Redirect:** This infrastructure enables production AI voice agents for outbound calling campaigns. The latency budget and reliability requirements are the same whether the use case is AI SDR qualification calls or customer success check-ins. The deployment pattern—WebSocket audio stream, dual KV cache, jitter buffer management—is the foundation for Zone 2 voice engagement at scale.

**Exercise hooks:**
- Easy: Write a script that generates synthetic latency measurements for 1000 frames and prints p50/p95/p99 percentiles.
- Medium: Simulate a jitter buffer: given a stream of frames with variable arrival times, implement a fixed-delay buffer that emits frames at constant rate and print the added latency vs. frame drop tradeoff.
- Hard: Implement a reconnection handler that detects WebSocket closure, retries with exponential backoff, and prints a session continuity report showing frames lost during reconnection.

---

## Beat 6: Assess

**Evaluate whether the practitioner can:**

1. **Explain** why half-duplex voice AI produces conversational artifacts that full-duplex S2S eliminates—specifically naming the dual-stream attention mechanism and overlap tokens, not just "lower latency."
2. **Compare** Moshi's architecture to a VAD-gate-then-generate pipeline in terms of latency budget, interrupt handling, and memory requirements.
3. **Calculate** whether a given frame rate and model inference time fits within a target end-to-end latency budget.
4. **Implement** a streaming inference loop with latency instrumentation.
5. **Evaluate** [CITATION NEEDED — concept: Hibiki's specific contribution to the full-duplex pipeline; assessment question pending clarification of Hibiki's role].

**Exercise hooks:**
- Easy: Given Mimi's 12.5 Hz frame rate and a target of 250ms total latency, calculate and print the maximum number of transformer forward passes allowed per frame.
- Medium: Write a function that takes two lists representing user and model token streams with timestamps, detects all overlap regions, and prints each overlap's start time, duration, and which party initiated the interruption.
- Hard: Design a latency budget breakdown for a production deployment (network RTT + encode + inference + decode + jitter buffer) and write a script that prints whether the total fits within 300ms given parameterized values for each stage.

---

## GTM Redirect Rules (Summary)

| Beat | Redirect Target |
|------|----------------|
| Use It | Zone 2 — AI Voice Agents cluster (outbound calling, objection handling, discovery calls) |
| Ship It | Zone 2 — Production voice engagement infrastructure (same deployment pattern for AI SDR and CS voice) |

If the practitioner is not building voice agents, this lesson is **foundational for Zone 2** — the dual-stream attention and real-time inference patterns transfer to any streaming multimodal system.