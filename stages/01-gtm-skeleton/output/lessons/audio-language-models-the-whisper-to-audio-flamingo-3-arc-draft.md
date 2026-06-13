# Audio-Language Models: The Whisper to Audio Flamingo 3 Arc

## Learning Objectives
1. Compare Whisper's encoder-decoder architecture to audio-language models that pair an audio encoder with an LLM decoder.
2. Trace the audio tokenization pipeline from raw waveform through Mel spectrogram to transformer tokens.
3. Implement a working transcription pipeline using Whisper.
4. Evaluate when transcription-only suffices versus when audio-language understanding is required.
5. Configure an audio processing pipeline for a GTM enrichment use case.

---

## Hook

The arc from Whisper (2022) to Audio Flamingo 3 (2024) tracks a specific architectural shift: from a supervised encoder-decoder trained to output transcripts, to an audio-language model that accepts audio as an input modality alongside arbitrary text instructions. This matters because GTM systems increasingly need to reason about conversations — what was objected to, what buying signals appeared, what competitors were mentioned — not just produce a transcript.

---

## Concept

Audio tokenization pipeline: raw waveform → short-time Fourier transform → Mel filterbank → log-mel spectrogram → patch extraction → linear projection → token sequence. Whisper trains an encoder-decoder transformer on 680K hours of supervised audio-transcript pairs across 99 languages. Audio-language models (Audio Flamingo 3, Qwen-Audio, SALMONN) replace the decoder with an LLM and align audio encoder outputs to the LLM's embedding space via a projection layer trained on audio-text instruction data. The architectural distinction: Whisper maps audio → text (one output type). Audio-language models map (audio, text instruction) → text (arbitrary output). The projection layer is the critical mechanism — contrastive pretraining aligns audio representations with text embeddings, then instruction tuning teaches the LLM to condition on audio tokens.

---

## Demo

Transcribe a sample audio file using the `whisper` Python package. Print transcript, detected language, and segment-level timing. Then demonstrate the boundary: ask a reasoning question about the audio content ("what objection did the prospect raise?") and show that Whisper cannot answer — it only transcribes. Walk through the architectural reason: the decoder is trained to predict the next transcript token, not to follow instructions about audio content.

---

## Use It

**GTM Cluster: Zone 2 Enrichment — sales call analysis and voice-of-customer signal extraction.**

Current production pattern: record call → transcribe with Whisper → run LLM over transcript to extract buying signals, objections, competitor mentions. The audio-language model arc collapses this to a single pass: audio + instruction in, structured signal out. Build a pipeline that processes audio files and extracts structured GTM signals using transcription followed by LLM extraction over the transcript. This is the same pattern used in enrichment workflows where raw data enters and structured signals exit.

---

## Ship It

Batch processing pattern for audio in production: chunk variable-length inputs with 30-second windows and overlap, manage GPU memory for concurrent requests, implement retry logic for noisy or corrupted files, design output schema for downstream enrichment tables. Cost tradeoff: Whisper-large-v3 accuracy versus Whisper-small/turbo latency for real-time use cases. For the audio-language model path: quantization (4-bit, 8-bit) and context window management for long conversations.

---

## Evaluate It

- **Easy:** Transcribe an audio file using Whisper and print the detected language, confidence score, and full transcript. Observable output confirms the model loaded and produced structured text.
- **Medium:** Build a batch pipeline that processes a directory of `.wav` files, transcribes each with Whisper, and writes a JSON file per audio containing transcript, duration, language, and segment count.
- **Hard:** Transcribe a set of simulated sales-call audio files, then run an LLM extraction pass over each transcript to output structured JSON: `{objection_type, buying_stage, competitor_mentions, next_steps}`. Compare the two-step pipeline's accuracy and latency against a single audio-language model pass (if available), documenting where transcription-only fails and where it suffices.