# Whisper — Architecture & Fine-Tuning

## Learning Objectives

1. Trace an audio signal through Whisper's encoder-decoder pipeline, from log-Mel spectrogram input through BPE token output.
2. Compare transcription quality and inference speed across Whisper model sizes on identical audio input.
3. Implement batch transcription of audio files with timestamped JSON output.
4. Configure and apply a LoRA adapter to shift Whisper's transcription behavior on domain-specific vocabulary.
5. Evaluate the tradeoffs between the 30-second window constraint and long-form transcription strategies.

## The Problem

Your prospects speak in meetings, on podcasts, in voicemails. Every sales call, every support recording, every voicemail drop contains buying signals — objections, intent phrases, competitor mentions, pricing questions. Most of that signal evaporates the moment the call ends. The text never reaches your CRM. The patterns never feed your scoring model.

Whisper, released by OpenAI in September 2022, was the first automatic speech recognition (ASR) model to ship as a commodity: paste audio, get text, 99 languages, runs on a laptop. By 2024 OpenAI had shipped Large-v3 and Turbo variants. By 2026, Whisper is the default baseline for everything from podcast transcription to voice assistants to YouTube subtitles. It is the entry point for any pipeline that converts conversational audio into structured text.

But Whisper is not a pipeline you can treat as a black box forever. Domain shift kills it. Feed it a sales call with heavy industry jargon and it will hallucinate product names. Feed it a voicemail with background noise and it will drop critical phrases. Feed it a 90-minute webinar and the 30-second window creates boundary artifacts that garble transitions between speakers. You need to know what the architecture actually does, how to feed it audio correctly, and when to reach for fine-tuning.

## The Concept

Whisper is a transformer