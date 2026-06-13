# Whisper — Architecture & Fine-Tuning

## 1. Hook

Your prospects speak in meetings, on podcasts, in voicemails. Most of that signal evaporates. Whisper converts audio to structured text — but only if you know which knobs change the output. This lesson dissects the architecture so you can tune it for real audio, not clean datasets.

## 2. Learn It

**Encoder-decoder transformer architecture.** Whisper encodes 30-second audio windows into log-Mel spectrograms, runs them through an encoder stack, then decodes token-by-token text with a language model head. The tokenizer handles 99 languages via byte-level BPE. Multitask training format uses special tokens (`<|transcribe|>`, `<|translate|>`, `<|lang|>`) to route behavior without architecture changes. Five model sizes scale from 39M to 1.5B parameters — each size changes the speed/accuracy tradeoff, not the mechanism.

**Fine-tuning mechanism.** LoRA adapters applied to attention projection matrices let you shift transcription behavior on domain-specific vocabulary (product names, industry jargon) without retraining the full model. The base weights stay frozen; only the low-rank matrices update.

## 3. See It

Load `openai/whisper-small`, pass a 30-second audio sample, and print the raw token output alongside the decoded transcription. Then inspect the log-Mel spectrogram shape to confirm the preprocessing pipeline. Finally, swap to `whisper-tiny` on the same audio and print theWER-like diff to observe the size/accuracy tradeoff directly.

## 4. Use It

**GTM redirect: Zone 1 — Signal capture from conversational audio.**

Sales calls, support recordings, and voicemail drops contain intent signals that never reach your CRM. Whisper converts those to text at scale. The GTM application: batch-transcribe call recordings, then pipe the output into your existing NLP pipeline for topic extraction, objection detection, or buying-signal scoring. [CITATION NEEDED — concept: audio signal pipeline for sales call analysis in GTM workflow]

## 5. Ship It

Write a script that loads `whisper-small`, transcribes a directory of `.wav` files, and writes timestamped JSON output. Then apply a HuggingFace `peft` LoRA adapter fine-tuned on 50 examples of domain-specific vocabulary and compare transcription accuracy before/after on a held-out test file. Print the diff.

**Exercise hooks:**
- *Easy:* Transcribe three audio files and print the raw JSON output with timestamps.
- *Medium:* Fine-tune a LoRA adapter on 20 labeled audio clips containing your company's product names. Measure the improvement.
- *Hard:* Build a batch pipeline that processes 100 recordings, detects language, routes to the correct Whisper model size based on language and audio quality, and outputs structured call summaries.

## 6. Push It

Whisper's 30-second window creates artifacts at boundaries during long-form transcription. The `word-level timestamps` head exists but is underdocumented in how it aligns tokens to audio frames — the mechanism relies on cross-attention weights extracted from the decoder, and this behavior shifts across model sizes. Streaming inference (partial transcriptions during recording) requires the `whisper-streaming` pattern, which chunks audio and manages context overlap — but this is a community pattern, not an official API. [CITATION NEEDED — concept: official streaming inference support for Whisper]

**Exercise hooks:**
- *Hard:* Implement word-level timestamp extraction from the cross-attention weights and validate against human-labeled boundaries. Report where alignment drifts.