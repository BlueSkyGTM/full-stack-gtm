# Phase 06 — Speech and Audio (quiz factory)

## Focus

Audio signal processing and speech ML: waveforms, spectrograms, MFCCs, ASR (CTC, attention-encoder-decoder, Whisper), TTS (Tacotron, VITS, Voicebox), voice activity detection, speaker diarisation, and audio generation.

## Scrape hints

- `docs/en.md`: signal processing formulas (STFT, mel filterbanks, MFCCs), loss functions (CTC, cross-entropy over text), architecture descriptions (encoder/decoder, conformer vs transformer)
- `code/main.py`: often pure-Python with `wave` / `struct` for raw audio — quiz what the code computes
- Vocabulary: `glossary/terms.md` for spectrogram, CTC, beam search, mel, phoneme, diarisation

## Style anchor

- No gold quiz in this phase yet — use `phases/07-transformers-deep-dive/15-attention-variants/quiz.json` for format reference
- Match the doc depth: pre = audio problem motivation, checks = algorithm mechanisms, post = lesson code or production tradeoff

## Common distractor patterns

- Confuse STFT (window-based FFT) with full-signal FFT
- Confuse CTC loss (blank token, no forced alignment) with attention cross-entropy loss
- Confuse mel filterbanks (perceptual frequency compression) with raw STFT magnitudes
- Conflate end-to-end ASR (Whisper, wav2vec 2.0) with two-stage pipeline (acoustic + language model)
- Mix TTS autoregressive (Tacotron 2) with flow-matching TTS (Voicebox, Matcha-TTS)

## Do not

- Import facts from other phases unless `docs/en.md` lists them as prerequisites.
- Ask the user questions — mark `blocked` in manifest instead.
