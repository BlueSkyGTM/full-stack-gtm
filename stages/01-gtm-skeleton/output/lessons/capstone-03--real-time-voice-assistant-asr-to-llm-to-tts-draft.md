# Capstone 03 — Real-Time Voice Assistant (ASR to LLM to TTS)

## Hook (Why This Pipeline Matters)

A voice assistant is three models stitched together under a latency budget. Speech must become text, text must become reasoning, reasoning must become speech — and the user notices every 100ms delay. This capstone wires ASR → LLM → TTS into a single event loop where streaming keeps the conversation feeling live instead of queued.

## Concept (Mechanisms & Architecture)

**The three-stage pipeline and its latency budget.** Audio arrives in frames (typically 100–250ms chunks). ASR converts frames to text in a streaming fashion — partial transcripts arrive mid-utterance, final transcripts land when the speaker pauses. The text feeds an LLM, which generates tokens incrementally. Those tokens route to TTS before the full response completes, so audio playback can start while the model is still generating. The critical mechanism is **partial result forwarding**: each stage begins work before the upstream stage finishes, trading completeness for responsiveness.

**Audio fundamentals that determine everything else.** Microphones capture PCM audio at a sample rate (commonly 16kHz for speech). Frames are chunked into fixed-size buffers (e.g., 4096 samples ≈ 256ms at 16kHz). Encoding (μ-law, Opus, raw PCM) affects both latency and quality. The pipeline must agree on format at every boundary — ASR expects a specific encoding, TTS produces a specific encoding, and mismatches cause either failures or garbled output.

**Streaming vs. request-response semantics.** A naïve pipeline treats each stage as a blocking HTTP call: record → wait → transcribe → wait → generate → wait → synthesize → wait → play. Each "wait" is 200–2000ms. Streaming collapses these waits by opening websocket or SSE connections and forwarding partial results. ASR sends interim transcripts. The LLM yields tokens one at a time. TTS begins synthesis on sentence boundaries mid-generation. The architecture is an event stream, not a batch job.

**Interruption handling — the hard problem.** When the user speaks while the assistant is talking, the pipeline must: stop TTS playback, discard pending LLM tokens, feed the new audio to ASR, and begin a new response. This requires a shared state object (or message bus) that all three stages check before producing output. Without it, the assistant talks over the user.

## Demo (Working Pipeline)

Build a voice assistant that accepts audio input from a file or microphone, processes it through ASR (OpenAI Whisper API), generates a response via LLM (OpenAI chat completions with streaming), and plays audio output via TTS (OpenAI TTS API). The demo shows:

- **Audio capture** using `sounddevice` into a ring buffer, chunked into frames
- **ASR streaming** via the Whisper API, printing partial transcripts as they arrive
- **LLM token streaming** with `openai.ChatCompletion.create(stream=True)`, forwarding tokens to a queue
- **TTS synthesis** consuming tokens from the queue, generating audio on sentence boundaries
- **Audio playback** of synthesized chunks using `sounddevice` again
- **Latency measurement** at each stage boundary, printed to console

The practitioner sees end-to-end latency numbers for each stage and the total pipeline, confirming where time is spent.

```python
import openai
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import re

SAMPLE_RATE = 16000
CHUNK_DURATION = 5
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.5

client = openai.OpenAI()

def record_audio():
    print("Recording... speak now. Silence for 1.5s will stop recording.")
    audio_chunks = []
    silence_start = None
    
    def callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy())
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback, blocksize=4096):
        while True:
            time.sleep(0.1)
            if len(audio_chunks) > 0:
                last_chunk = audio_chunks[-1]
                rms = np.sqrt(np.mean(last_chunk**2))
                if rms < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        break
                else:
                    silence_start = None
    
    audio_data = np.concatenate(audio_chunks, axis=0)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return audio_int16.tobytes()

def transcribe(audio_bytes):
    print("[ASR] Sending audio to Whisper...")
    t0 = time.time()
    import io
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    
    import wave
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)
    wav_buffer.seek(0)
    wav_buffer.name = "recording.wav"
    
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_buffer,
        response_format="text"
    )
    elapsed = time.time() - t0
    print(f"[ASR] Transcription ({elapsed:.2f}s): {response}")
    return response

def generate_response(transcript):
    print("[LLM] Generating streaming response...")
    t0 = time.time()
    token_queue = queue.Queue()
    
    def stream_tokens():
        full_response = ""
        buffer = ""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise voice assistant. Keep responses under 3 sentences."},
                {"role": "user", "content": transcript}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                buffer += token
                if buffer.endswith(('.', '!', '?', '\n')):
                    token_queue.put(("sentence", buffer.strip()))
                    buffer = ""
        if buffer.strip():
            token_queue.put(("sentence", buffer.strip()))
        token_queue.put(("done", full_response))
    
    threading.Thread(target=stream_tokens, daemon=True).start()
    return token_queue

def speak_response(token_queue):
    print("[TTS] Synthesizing speech from streamed sentences...")
    total_t0 = time.time()
    
    while True:
        msg_type, content = token_queue.get()
        if msg_type == "done":
            total_elapsed = time.time() - total_t0
            print(f"[TTS] Full response spoken in {total_elapsed:.2f}s")
            print(f"[TTS] Complete text: {content}")
            break
        
        t0 = time.time()
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=content,
            response_format="pcm"
        )
        audio_bytes = response.content
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        elapsed = time.time() - t0
        print(f"  [TTS] Synthesized ({elapsed:.2f}s): {content}")
        
        sd.play(audio_np, samplerate=24000)
        sd.wait()

def main():
    print("=== Voice Assistant Pipeline ===")
    print("Measuring latency at each stage.\n")
    
    pipeline_start = time.time()
    
    audio_bytes = record_audio()
    rec_elapsed = time.time() - pipeline_start
    print(f"[Record] Captured audio in {rec_elapsed:.2f}s\n")
    
    transcript = transcribe(audio_bytes)
    
    if not transcript.strip():
        print("No speech detected. Exiting.")
        return
    
    token_queue = generate_response(transcript)
    speak_response(token_queue)
    
    total_elapsed = time.time() - pipeline_start
    print(f"\n[Total Pipeline] {total_elapsed:.2f}s from recording start to final word spoken.")

if __name__ == "__main__":
    main()
```

## Exercise (Practice Hooks)

- **Easy**: Modify the system prompt to change the assistant persona. Measure whether persona length affects LLM time-to-first-token. Print the latency delta.
- **Medium**: Replace the silence-based stopping condition with a fixed-duration recording (3 seconds). Compare ASR accuracy between silence-gated and fixed-duration capture. Print both transcripts side by side.
- **Hard**: Implement barge-in (interruption). While TTS audio is playing, monitor the microphone input level. If RMS exceeds a threshold for 300ms, stop playback, discard the remaining token queue, and begin a new ASR cycle. Print `"[BARGE-IN DETECTED]"` when this triggers.

## Use It (GTM Redirect)

This pipeline is the architecture behind **AI voice agents for outbound sales and inbound qualification** — the same pattern used in real-time SDR call agents. The ASR→LLM→TTS event loop maps directly to Zone 2 (Engagement) in the GTM topic map: the assistant engages a prospect, processes their objection via LLM, and responds in real-time. The sentence-boundary forwarding mechanism is what separates a conversational agent from a walkie-talkie. The latency measurement exercise maps to the metric GTM teams track: **time-to-first-audio**, which determines whether a prospect stays on the call or hangs up.

[CITATION NEEDED — concept: GTM Zone 2 voice agent engagement patterns]

## Ship It (Production Considerations)

**From capstone to production voice agent.** The demo uses OpenAI's hosted APIs for all three stages — acceptable for prototyping, insufficient for production. The migration path:

1. **ASR**: Swap Whisper API for a streaming ASR service (Deepgram, AssemblyAI) that emits interim transcripts over websockets. Latency drops from 500ms+ to under 200ms for final transcripts, and interim results arrive in under 50ms.

2. **LLM**: Replace the generic `gpt-4o-mini` call with a fine-tuned model or prompt chain that encodes your product's sales playbook, objection handling, and qualification criteria. Add a **turn-taking classifier** that predicts whether the speaker has finished — this replaces the silence heuristic with a semantic decision.

3. **TTS**: Use a streaming TTS API (ElevenLabs, PlayHT) that accepts token-by-token input and produces audio chunks without waiting for a full sentence. This eliminates the largest source of latency in the demo pipeline.

4. **Transport**: Wrap the pipeline in a WebSocket server (FastAPI + websockets) so callers connect via SIP/PSTN gateway (Twilio, Telnyx). Each call gets its own pipeline instance with shared state for interruption handling.

5. **Observability**: Log every stage transition with timestamps. The key production metric is **conversational latency**: the gap between the user stopping speech and the assistant starting speech. Target under 800ms. If it exceeds that, the conversation feels unnatural and prospects disengage.

**GTM redirect**: Production voice agents at scale require the same orchestration discipline as Zone 3 (Conversion) automation workflows — every millisecond of latency maps to a measurable drop in conversion rate. The pipeline architecture you built here is the foundation; the production hardening is where GTM impact lives.

[CITATION NEEDED — concept: voice agent latency benchmarks and conversion rate correlation]