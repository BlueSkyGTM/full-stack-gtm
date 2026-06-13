# Voice Agents: Pipecat and LiveKit

## Hook

Real-time voice agents must solve a hard latency budget: speech-to-text inference, LLM token generation, text-to-speech synthesis, and audio transport — all within ~300ms to avoid sounding robotic. This lesson breaks down the pipeline architecture that orchestrates those stages.

## Concept

Explain the mechanism of a voice agent pipeline: a continuous audio stream enters via WebRTC transport (LiveKit's role), gets segmented by voice activity detection, transcribed by STT, processed by an LLM, synthesized back to audio by TTS, and returned over the same transport — with interruption handling that cancels in-flight TTS when the user speaks again. Then name the tools: LiveKit provides the real-time audio transport layer (rooms, tracks, WebRTC negotiation); Pipecat provides the pipeline orchestration (connecting STT → LLM → TTS with buffering, turn-taking, and endpointing).

## Code

Build a minimal voice bot that joins a LiveKit room, listens for speech, sends transcription to an LLM, and streams audio responses back. Two scripts: one that sets up the Pipecat pipeline with a LiveKit transport, and one that demonstrates the pipeline running with observable console output showing each stage (VAD triggered → STT result → LLM tokens → TTS audio bytes sent). Include a local-only test that confirms the pipeline wiring without requiring a live microphone.

## Use It

GTM cluster: Zone 1 (Prospecting / Outbound). Voice agents implement automated qualification calls and inbound lead routing. Map the pipeline stages to a GTM workflow: STT captures prospect responses, LLM evaluates fit against ICP criteria, TTS delivers the next question or hands off to a human AE. Exercise hooks: easy — configure a Pipecat bot to ask three qualification questions; medium — add branching logic based on transcript content; hard — implement real-time interruption when the prospect says a trigger phrase indicating purchase intent.

## Ship It

Deploy the voice agent to a server capable of maintaining a persistent LiveKit room. Cover the production concerns: ambient noise rejection tuning, latency profiling per pipeline stage, graceful degradation when STT confidence drops, and call termination logic. Exercise hooks: easy — deploy a Pipecat bot to a cloud VM and join a test call; medium — add logging that measures latency at each pipeline stage and prints a budget report; hard — implement a fallback to a simpler TTS provider when primary synthesis latency exceeds a threshold.

## Evaluate

Assess whether the practitioner can: explain why WebRTC is used instead of HTTP for audio transport; diagram the STT → LLM → TTS pipeline and identify which stage dominates the latency budget; configure a Pipecat pipeline with a LiveKit transport; implement interruption handling; and map voice agent output to a GTM qualification workflow. Quiz questions grounded in pipeline mechanics and observable behavior, not trivia.