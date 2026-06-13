# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

## Learning Objectives

1. Compare the audio encoding architectures of Qwen2.5-Omni, Audio Flamingo, and GPT-4o Audio.
2. Implement audio-to-text inference using Qwen2.5-Omni via Hugging Face transformers.
3. Evaluate model outputs on speech versus non-speech audio inputs.
4. Configure an audio ingestion pipeline that feeds structured output into a GTM enrichment workflow.

---

## Beat 1: Hook

Most GTM teams treat call recordings as write-only archives. Audio-language models turn those recordings into queryable, structured signals — emotion, objection patterns, competitive mentions — without a transcription intermediary. The question is whether the open-source options (Qwen2.5-Omni, Audio Flamingo) are viable alternatives to GPT-4o Audio for production pipelines.

---

## Beat 2: Concept

### The Mechanism: Audio Tokenization Meets Language Decoding

Audio-language models share a common architecture: an audio encoder compresses raw waveform or spectrogram data into a sequence of dense vectors, which a language model then decodes alongside (or instead of) text tokens.

**Audio Flamingo** — Perceiver-resampler architecture. Audio encoder (typically a pretrained CLAP or Whisper backbone) produces variable-length embeddings. A perceiver resampler compresses these into a fixed number of "audio tokens" that get prepended or interleaved with text tokens before entering the LLM. Few-shot audio QA is the design goal. [CITATION NEEDED — concept: Audio Flamingo perceiver resampler token count and compression ratio]

**Qwen2.5-Omni** — Single model, five modalities. Audio input passes through an encoder that produces tokens in the same embedding space as text. The model processes interleaved audio-text sequences natively — no adapter module between encoder and LLM backbone. Audio output is also supported via a codec-based decoder. [CITATION NEEDED — concept: Qwen2.5-Omni audio encoder architecture details — whether Whisper-based or custom]

**GPT-4o Audio** — Undocumented architecture. Observable behavior: accepts audio input, returns text or audio output. Latency profiles suggest end-to-end training rather than a cascaded ASR→LLM pipeline. No public weights, no architecture paper. The mechanism is inferred from API behavior.

### Key Architectural Comparison

| Dimension | Audio Flamingo | Qwen2.5-Omni | GPT-4o Audio |
|---|---|---|---|
| Audio encoder | External (CLAP/Whisper) | Integrated | Unknown |
| Token bridge | Perceiver resampler | Shared embedding space | Unknown |
| Audio output | No | Yes (codec decoder) | Yes |
| Open weights | Yes | Yes | No |
| Non-speech audio | Limited | Yes (music, environment) | Yes |

### Why This Matters for Practitioners

The perceiver-resampler approach (Audio Flamingo) adds a compression bottleneck — information loss is architectural. The shared-embedding approach (Qwen2.5-Omni) preserves more audio detail but demands more compute. GPT-4o Audio's behavior suggests shared embeddings, but without weights this is speculation.

---

## Beat 3: Demo

Run Qwen2.5-Omni inference on a sample audio file and extract structured output. Demonstrate both speech transcription and non-speech audio description.

### Code Example 1: Basic Audio Inference with Qwen2.5-Omni

```python
import torch
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info

model_id = "Qwen/Qwen2.5-Omni-7B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Sample_audio_file.wav"},
            {"type": "text", "text": "Describe what you hear in this audio clip. Provide: 1) transcription of any speech, 2) description of non-speech sounds, 3) overall audio quality."},
        ],
    }
]

audios = [process_mm_info(msg["content"]) for msg in messages]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, audios=audios, return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
output_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print("=== Qwen2.5-Omni Audio Output ===")
print(output_text)
```

### Code Example 2: Batch Processing Multiple Audio Files with Structured Extraction

```python
import json
import torch
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info

model_id = "Qwen/Qwen2.5-Omni-7B"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

audio_files = [
    "https://upload.wikimedia.org/wikipedia/commons/4/4f/Sample_audio_file.wav",
]

extraction_prompt = """Extract the following from this audio as JSON:
{
  "speech_detected": true/false,
  "transcription": "exact words spoken or null",
  "non_speech_sounds": ["list of sounds"],
  "language": "detected language code",
  "duration_estimate": "short/medium/long"
}
Return only valid JSON."""

results = []
for audio_url in audio_files:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_url},
                {"type": "text", "text": extraction_prompt},
            ],
        }
    ]
    
    audios = [process_mm_info(msg["content"]) for msg in messages]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audios=audios, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    results.append({"audio_url": audio_url, "output": output_text})
    print(f"Processed: {audio_url}")
    print(f"Output: {output_text}")
    print("---")

print(f"\nTotal files processed: {len(results)}")
```

---

## Beat 4: Guided Exercise

### Easy
Load Qwen2.5-Omni and run inference on a provided WAV file containing a short spoken sentence. Print the raw model output.

### Medium
Build a function that accepts a local audio file path, runs Qwen2.5-Omni inference, and returns a Python dictionary with keys: `transcription`, `sounds_detected`, `sentiment_hint`. Handle the case where no speech is detected.

### Hard
Compare outputs from Qwen2.5-Omni and a Whisper-based transcription pipeline on the same audio file. Measure: (1) transcription accuracy, (2) ability to describe non-speech audio, (3) inference time. Print a side-by-side comparison table.

---

## Beat 5: Use It

### GTM Redirect: Enrichment & Signal Extraction (Zone 03 — Enrich)

Audio-language models transform call recordings, voicemails, and demo recordings into structured data without requiring a separate ASR step followed by an LLM pass.

**Concrete application:** Process Gong/Chorus call recordings through Qwen2.5-Omni to extract objection patterns, competitor mentions, and buying signal strength in a single inference pass. This feeds directly into the Clay waterfall as an enrichment step — the audio model output populates fields that downstream scoring uses.

**Why audio-language models over ASR→LLM pipelines:** Non-speech audio carries signal. Tone, pause patterns, background noise (competitive radio ads, call center sounds) are lost in pure transcription. Audio-language models with shared embeddings preserve this information.

**Implementation pattern:**
1. Export call recordings from conversation intelligence platform
2. Run batch inference through Qwen2.5-Omni with structured extraction prompts
3. Normalize output to JSON schema matching your CRM enrichment fields
4. Push to Clay via webhook or API for waterfall enrichment

[CITATION NEEDED — concept: Clay webhook ingestion format for audio-derived enrichment data]

---

## Beat 6: Ship It

Build an audio enrichment pipeline that processes a directory of call recordings and outputs a CSV ready for CRM import. The pipeline must handle failures gracefully (corrupt audio, no speech detected, model timeout) and log each file's processing status.

### Requirements

- Accept a directory path containing audio files (WAV, MP3, M4A)
- Process each file through Qwen2.5-Omni with a GTM-focused extraction prompt
- Output CSV columns: `file_name`, `transcription`, `objections_detected`, `competitor_mentions`, `next_step_suggestion`, `processing_status`
- Log processing time per file
- Skip files that fail and record the error
- Print a summary: total files, successful, failed, average processing time

This pipeline maps to Zone 03 (Enrich) in the GTM content map — the output CSV feeds directly into Clay or any enrichment waterfall as a signal source.