# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

## Learning Objectives

1. Compare the audio encoding architectures of Qwen2.5-Omni, Audio Flamingo, and GPT-4o Audio across encoder type, token bridge, and output modality.
2. Implement audio-to-text inference using Qwen2.5-Omni via Hugging Face `transformers`, processing both speech and non-speech inputs.
3. Evaluate model responses on three audio categories — speech, environmental sound, and synthetic tone — to identify where each architecture's compression strategy loses information.
4. Configure an audio ingestion pipeline that extracts structured JSON from recordings and feeds those signals into a GTM enrichment workflow.

---

## The Problem

You have 5 seconds of audio: a dog barks, someone yells "stop!", then silence. Useful questions span multiple axes that traditional ASR cannot answer.