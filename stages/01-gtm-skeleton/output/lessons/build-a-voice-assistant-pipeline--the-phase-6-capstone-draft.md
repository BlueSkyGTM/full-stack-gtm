# Build a Voice Assistant Pipeline — The Phase 6 Capstone

## Hook (Beat 1)

You've built agents that read text and write text. Now you wire those agents into a real-time audio loop: speech in, reasoning in the middle, speech out. This is the integration capstone — it pulls from every prior phase and exposes every latent latency bug you've been ignoring.

## Concept (Beat 2)

A voice assistant is three queues connected by two models. Speech-to-text converts incoming audio frames into a transcript stream. An LLM consumes that transcript and generates a response token-by-token. Text-to-speech converts those tokens back into audio before the human gets impatient and hangs up. The hard part is not any single model — it's managing the buffer between them so that latency stays under human attention thresholds (~300ms for perceived responsiveness). This beat covers the pipeline topology, the buffering strategy at each seam, and the failure modes that break the illusion of conversation.

## Demo (Beat 3)

A single Python script that accepts a WAV file, transcribes it with a local Whisper model, pipes the transcript to Claude for a response, and synthesizes the answer back to audio. Every stage prints its own latency. Output is a WAV file you can play, plus a printed timing breakdown showing where milliseconds accumulate.

*Exercise hooks:*
- **Easy:** Run the provided script against the included test WAV. Report which stage consumes the most wall-clock time.
- **Medium:** Swap the local Whisper endpoint for a remote API call. Measure the latency delta.
- **Hard:** Add streaming output so that TTS begins before the LLM has finished generating the full response.

## Lab (Beat 4)

You extend the demo into a turn-taking loop. The pipeline accepts multiple audio inputs in sequence, maintains a conversation history across turns, and exits on a silence threshold or explicit stop word. You must implement the buffering logic that decides when the speaker has finished a thought — voice activity detection or a simple energy threshold — before triggering the LLM call.

*Exercise hooks:*
- **Easy:** Add a second turn to the conversation using a second WAV file. Confirm history is preserved.
- **Medium:** Implement energy-based voice activity detection so the pipeline auto-detects when the user stops speaking.
- **Hard:** Interruption handling — if new audio arrives while the TTS is still playing, cut playback and re-process the new input.

## Use It (Beat 5)

This pipeline is the mechanism behind conversational voice agents used in outbound qualification and inbound routing. In the GTM context, the pattern maps to **Zone 3 — Automated Outreach & Enrichment**, specifically voice-based SDR workflows where a synthetic agent qualifies a lead in real time. [CITATION NEEDED — concept: voice SDR agent pipeline architecture] The LLM generates context-appropriate questions, the STT/TTS loop handles the conversation surface, and the latency budget determines whether the prospect perceives a human or a machine. Any GTM practitioner deploying Clay waterfall enrichment into a voice channel is wiring the same buffer-management pattern covered here — the data source changes, the queue discipline does not.

## Ship It (Beat 6)

Package the pipeline as a CLI tool that accepts a directory of WAV files (simulating a batch of lead responses), runs each through the assistant, and writes a JSON log of transcripts, responses, and per-stage latencies. This output is your capstone artifact — auditable, reproducible, and measurable. Include a `README.md` with latency targets and known failure modes. Submit the repo.

*Exercise hooks:*
- **Easy:** Ship the CLI with working defaults and one sample WAV. Verify JSON output matches the expected schema.
- **Medium:** Add a `--latency-budget` flag that rejects any run where total round-trip exceeds a user-specified threshold in milliseconds.
- **Hard:** Deploy the pipeline as a local HTTP server that accepts POSTed audio and returns synthesized speech, with a `/health` endpoint that reports average latency over the last 10 requests.