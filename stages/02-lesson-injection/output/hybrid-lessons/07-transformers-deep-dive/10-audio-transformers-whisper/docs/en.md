# Audio Transformers — Whisper Architecture

## Learning Objectives

- Implement a Whisper transcription pipeline that converts raw audio into timestamped, language-detected text
- Trace the encoder-decoder flow from log-Mel spectrogram through convolutional stem, encoder self-attention, decoder cross-attention, and autoregressive token generation
- Compare Whisper's output behavior under speech versus silence conditions, documenting hallucination patterns
- Build a batch call-processing pipeline that emits structured JSON suitable for downstream CRM enrichment
- Evaluate confidence-based filtering strategies for suppressing low-quality transcription segments before they enter GTM workflows

## The Problem

Most buyer-relevant signal in B2B — objections, pricing questions, competitor mentions, next-step commitments — lives in recorded conversations. Sales calls, discovery meetings, voicemails, demo recordings. That audio is opaque to every downstream system: your CRM, your scoring model, your enrichment pipeline. None of them can query a WAV file. Before that signal becomes actionable, it has to become text.

Traditional automatic speech recognition (ASR) made this expensive. Systems like wav2vec 2.0 and HuBERT used self-supervised pretraining on clean academic corpora, then required task-specific fine-tuning heads. Quality was high in-domain, but distribution shift — a new accent, background noise, a language with limited training data — degraded output fast. Multilingual support meant training separate models per language family. For a GTM team processing calls across regions, that meant multiple inference pipelines, inconsistent formatting, and maintenance overhead per language.

Whisper (OpenAI, Radford et al. 2022) took a different path. Three bets: train on 680,000 hours of weakly-labeled audio scraped from the internet across 97 languages — no clean corpus, no phoneme labels. Use a single decoder trained jointly on transcription, translation, voice activity detection, language ID, and timestamping via special task tokens. And keep the architecture a standard encoder-decoder transformer — encoder consumes log-Mel spectrograms, decoder produces text tokens autoregressively. No vocoder, no CTC, no HMM. The encoder-decoder structure is what enables the multi-task format: the decoder can be conditioned to transcribe, translate, or detect language in a single forward pass by changing the prefix tokens it receives.

## The Concept

Whisper treats audio as an image. Not metaphorically — literally. The first step is converting the raw audio waveform into a log-Mel spectrogram, a 2D representation where one axis is time and the other is frequency (binned on the Mel scale, which approximates human auditory perception). The encoder then processes this spectrogram the same way a Vision Transformer processes an image: as a grid of patches projected into embedding space. If you completed the ViT lesson, the pattern is identical — the only difference is the input modality.

Here is the full pipeline:

```mermaid
flowchart LR
    A["Raw Audio<br/>16 kHz mono"] --> B["Resample + Pad/Truncate<br/>to 30 seconds"]
    B --> C["Log-Mel Spectrogram<br/>80 mel bins × ~3000 frames"]
    C --> D["Conv1D Stem<br/>2 layers, stride 2<br/>→ 1500 frames"]
    D --> E["Encoder<br/>32 layers, self-attention<br/>→ acoustic embeddings"]
    E -.->|cross-attention| G
    F["Task Tokens<br/>⟨lang⟩⟨task⟩"] --> G["Decoder<br/>32 layers, self-attention<br/>+ cross-attention"]
    G --> H["Text Tokens<br/>autoregressive"]
    H --> I["Detokenize<br/>→ transcript + timestamps"]
```

### Step 1 — Resample and Window

Audio is resampled to 16 kHz mono — Whisper does not operate on stereo or non-16kHz input without preprocessing. The audio is then clipped or padded to exactly 30 seconds. This 30-second window is a hard architectural constraint, not a recommendation. The positional encodings in the encoder are sized for 1,500 tokens (after the convolutional stem halves the 3,000 spectrogram frames twice). Audio longer than 30 seconds requires external sliding-window logic that stitches segments together. Audio shorter than 30 seconds gets padded with silence — which matters, because that silence has consequences we will see in Build It.

### Step 2 — Log-Mel Spectrogram

The waveform is converted to a spectrogram using a Short-Time Fourier Transform (STFT) with a 25ms window and 10ms stride, then mapped onto 80 Mel frequency bins. The Mel scale spaces frequencies according to human hearing sensitivity — more resolution at low frequencies where speech carries phonemic information, less at high frequencies. The values are log-scaled because loudness perception is logarithmic. The output is a 80 × 3000 matrix for 30 seconds of audio. This is the "image" Whisper sees.

### Step 3 — Convolutional Stem

Two Conv1D layers, each with filter width 3 and stride 2, process the spectrogram before it reaches the transformer encoder. Each stride-2 convolution halves the sequence length. Two layers take 3,000 frames down to 1,500. This is a lightweight downsampling step — it reduces the number of tokens the encoder's self-attention must attend over, cutting attention complexity by roughly 4x (from O(3000²) to O(1500²)) without discarding much acoustic information. The conv stem also learns local frequency patterns that are useful across all languages.

### Step 4 — Encoder

The encoder is a standard transformer encoder — for the large model, 32 layers of multi-head self-attention with GELU feed-forward networks. Sinusoidal positional encodings are added to the conv stem output so the encoder knows which timestep each token represents. Self-attention lets every acoustic frame attend to every other frame, capturing long-range dependencies like the relationship between a phoneme at second 3 and a coarticulation effect at second 12. The output is a sequence of 1,500 acoustic embedding vectors.

### Step 5 — Decoder with Cross-Attention

The decoder is where the multi-task design lives. It is a 32-layer autoregressive transformer that generates text one token at a time. But before it generates any text, it receives a prefix of special tokens that tell it what to do:

- `<|en|>` — the detected source language
- `<|transcribe|>` or `<|translate|>` — the task
- `<|notimestamps|>` or `<|0.00|>` — whether to produce timestamps

For transcription, the decoder generates text in the source language. For translation, it generates English text regardless of source language. The decoder's cross-attention layers attend to the encoder's acoustic embeddings — this is the bridge between what was heard and what is being said. The decoder predicts the next token conditioned on both the text it has already generated (causal self-attention) and the acoustic representation (cross-attention).

### Step 6 — Silence and Hallucination

When the encoder receives padded silence or low-signal audio, the acoustic embeddings carry minimal information. But the decoder still generates tokens autoregressively — it cannot output nothing. Under low-signal conditions, the decoder falls back on its training distribution, which was internet-scraped audio that often had background music, ambient speech, or narration. The result is hallucinated text: repeated phrases, phantom transcriptions of nonexistent speech, or loops where the decoder generates the same phrase repeatedly. This is not a bug in the traditional sense — it is the expected behavior of a language model conditioned on uninformative input. For GTM pipelines, this means silence at the end of voicemails or hold music during calls can produce fabricated transcript content that, if ingested into a CRM, creates false signal.

## Build It

Let us run Whisper end-to-end and observe every output the model produces — language detection, transcription, and token-level timestamps. Then we will feed it silence and watch what happens.

```python
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai-whisper", "gtts"])
```

```python
from gtts import gTTS
import whisper
import json
import numpy as np
from scipy.io import wavfile

tts = gTTS(
    text="Hey Sarah, this is Mike from Acme Corp. I wanted to follow up on our pricing discussion from last week. Our enterprise plan starts at twelve thousand per year. Can we schedule a call next Tuesday to go over the details? Thanks.",
    lang="en",
    slow=False,
)
tts.save("sample_call.mp3")

model = whisper.load_model("base")

result = model.transcribe(
    "sample_call.mp3",
    task="transcribe",
    word_timestamps=True,
)

print("=== DETECTED LANGUAGE ===")
print(f"Language: {result['language']}")
print(f"Probability: {result.get('language_probability', 'N/A')}")

print("\n=== FULL TRANSCRIPT ===")
print(result["text"])

print("\n=== SEGMENTS WITH TIMESTAMPS ===")
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

print("\n=== WORD-LEVEL TIMESTAMPS (first 10 words) ===")
for seg in result["segments"][:1]:
    for word in seg.get("words", [])[:10]:
        print(f"  {word['start']:.2f}s - {word['end']:.2f}s : {word['word']}")

print("\n=== RAW TOKEN IDS (first 20) ===")
print(result["segments"][0]["tokens"][:20] if result["segments"] else "No segments")
```

When you run this, you will see the language auto-detected as English, the full transcript, segment boundaries with timestamps, and word-level timing. The token IDs in the raw output include the special task tokens prepended by the decoder prefix.

Now let us observe the hallucination behavior under silence:

```python
sample_rate = 16000
duration = 10
silence = np.zeros(sample_rate * duration, dtype=np.float32)
wavfile.write("silence.wav", sample_rate, silence)

result_silence = model.transcribe(
    "silence.wav",
    task="transcribe",
)

print("=== SILENT AUDIO TRANSCRIPT ===")
print(f"Text: '{result_silence['text']}'")
print(f"Segments: {len(result_silence['segments'])}")
for seg in result_silence["segments"]:
    print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] '{seg['text']}'")

result_translate = model.transcribe(
    "sample_call.mp3",
    task="translate",
)

print("\n=== TRANSLATION TASK (to English) ===")
print(f"Detected language: {result_translate['language']}")
print(f"Translated text: {result_translate['text']}")
```

The silent audio will produce non-empty output. Depending on the model size and random seed, you will see repeated phrases like "Thank you." or "Thanks for watching!" — text that exists nowhere in the audio. This is the decoder's prior taking over when the encoder provides no meaningful signal. The translation task output demonstrates the multi-task format: the same model, same weights, different prefix token, English output regardless of input language.

## Use It

The structured-field extraction we are about to build is the front-end of Zone 3 — Signal Processing & Enrichment. In a GTM stack, conversation audio is raw signal that must be converted into structured fields before any enrichment, scoring, or routing logic can act on it. [CITATION NEEDED — concept: Whisper-based call analytics pipeline in GTM enrichment workflows] The same way Clay's waterfall enriches a domain into a company profile, Whisper enriches a WAV file into objections, pricing signals, and next steps. The mechanism is transcription followed by pattern extraction — the AI concept is the encoder-decoder's cross-attention producing text from acoustic features, and the GTM application is turning that text into CRM fields.

Here is a practical extraction pipeline that takes a transcribed sales call and pulls structured fields:

```python
import re
import json
from gtts import gTTS
import whisper

call_script = """
Hey Sarah, thanks for taking the call. So looking at what we discussed last time, the main concern was around implementation timeline. You mentioned your team needs this live by Q3. I think we can work with that. On pricing, the enterprise tier is fifteen thousand per year with a two-year commitment. If you need month-to-month, it goes up to eighteen hundred per month. I know you are also looking at Competitor X — their pricing is similar but they do not have the API access you need. The next step would be to get you a security questionnaire response by Friday. Can you confirm who else needs to sign off on the legal side?
"""

tts = gTTS(text=call_script, lang="en", slow=False)
tts.save("sales_call.mp3")

model = whisper.load_model("base")
result = model.transcribe("sales_call.mp3", task="transcribe")
transcript = result["text"]

print("=== TRANSCRIPT ===")
print(transcript)

pricing_patterns = [
    (r'\$?(\d[\d,]*)\s*(?:per\s*year|annually|/year)', "annual_price"),
    (r'\$?(\d[\d,]*)\s*(?:per\s*month|monthly|/mo)', "monthly_price"),
    (r'(\d[\d,]*)\s*(?:per\s*year|annually)', "annual_price"),
]

timeline_patterns = [
    r'\b(Q[1-4])\b',
    r'\bby\s+(\w+day)\b',
    r'\bby\s+(Friday|Monday|Tuesday|Wednesday|Thursday|Saturday|Sunday)\b',
    r'\bnext\s+(\w+)\b',
]

competitor_patterns = [
    r'\b(?:looking at|evaluating|considering|comparing against)\s+([A-Z][a-zA-Z\s]+?)(?:\s+[-—,]|$)',
]

next_step_patterns = [
    r'(?:next step|next steps|action item|follow.?up)[:\s]+(.*?)(?:\.|$)',
    r'(?:I will|I\'ll|we will|we\'ll)\s+(.*?)(?:\.|$)',
    r'(?:can you|could you)\s+(.*?)(?:\?|$)',
]

objection_keywords = [
    "concern", "worried", "issue", "problem", "hesitant",
    "expensive", "budget", "timeline", "too long", "complicated",
]

extracted = {
    "pricing": {},
    "timeline": [],
    "competitors": [],
    "next_steps": [],
    "objections": [],
}

for pattern, price_type in pricing_patterns:
    matches = re.findall(pattern, transcript, re.IGNORECASE)
    if matches:
        extracted["pricing"][price_type] = matches

for pattern in timeline_patterns:
    matches = re.findall(pattern, transcript, re.IGNORECASE)
    extracted["timeline"].extend(matches)

for pattern in competitor_patterns:
    matches = re.findall(pattern, transcript)
    extracted["competitors"].extend(matches)

for pattern in next_step_patterns:
    matches = re.findall(pattern, transcript, re.IGNORECASE)
    extracted["next_steps"].extend(matches)

sentences = re.split(r'[.!?]+', transcript)
for sent in sentences:
    for keyword in objection_keywords:
        if keyword.lower() in sent.lower():
            extracted["objections"].append(sent.strip())
            break

extracted["pricing"] = {k: v for k, v in extracted["pricing"].items() if v}
extracted = {k: v for k, v in extracted.items() if v}

print("\n=== EXTRACTED STRUCTURED FIELDS ===")
print(json.dumps(extracted, indent=2))

crm_record = {
    "call_id": "call_2024_001",
    "transcript": transcript,
    "language": result["language"],
    "duration_seconds": result["segments"][-1]["end"] if result["segments"] else 0,
    "extracted_fields": extracted,
}

print("\n=== CRM-READY RECORD ===")
print(json.dumps(crm_record, indent=2))
```

The output is a structured record with pricing details, timeline references, competitor mentions, next steps, and objection-flagged sentences. This is the raw material that feeds into deal scoring, routing rules, and account-based marketing orchestration. The encoder-decoder's cross-attention produced the acoustic-to-text mapping; the regex layer on top produces the text-to-field mapping. Both are signal transformations — one from waveform to language, one from language to CRM schema.

## Ship It

A production call-processing pipeline needs to handle a directory of recordings, fail gracefully on corrupt files, and write output that downstream systems can consume. This is the same pattern as a batch enrichment waterfall: process many inputs, log what succeeded, log what failed, write structured output.

```python
import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

import whisper

INTENT_PATTERNS = {
    "pricing_discussion": [
        r"\b(?:price|pricing|cost|budget|expensive|afford)\b",
        r"\$\d+",
        r"\d+\s*(?:per\s*month|per\s*year|/mo|/yr|annually|monthly)",
    ],
    "competitor_mentioned": [
        r"\b(?:competitor|alternative|comparing|evaluating|looking at)\b",
        r"\b(?:vs\.?|versus)\b",
    ],
    "objection_raised": [
        r"\b(?:concern|worried|issue|problem|hesitant|risk)\b",
        r"\b(?:not sure|unsure|skeptical)\b",
    ],
    "next_step_committed": [
        r"\b(?:next step|follow.?up|action item|send over|get back)\b",
        r"\b(?:schedule|calendar|book)\b",
    ],
    "decision_maker_referenced": [
        r"\b(?:CEO|CTO|CFO|VP|director|head of|my boss|sign.?off|approval)\b",
    ],
}

ENTITY_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "date": r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b',
    "money": r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:per\s*)?(?:year|month|day|week|annually|monthly))?',
    "url": r'https?://[^\s]+',
}

def classify_intents(transcript):
    intents = {}
    transcript_lower = transcript.lower()
    for intent_name, patterns in INTENT_PATTERNS.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, transcript, re.IGNORECASE)
            matches.extend(found)
        if matches:
            intents[intent_name] = {
                "match_count": len(matches),
                "examples": matches[:5],
            }
    return intents

def extract_entities(transcript):
    entities = {}
    for entity_type, pattern in ENTITY_PATTERNS.items():
        matches = re.findall(pattern, transcript)
        if matches:
            entities[entity_type] = list(set(matches))
    return entities

def transcribe_file(model, file_path):
    try:
        result = model.transcribe(
            str(file_path),
            task="transcribe",
            word_timestamps=False,
        )
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        transcript_text = result["text"].strip()
        intents = classify_intents(transcript_text)
        entities = extract_entities(transcript_text)

        return {
            "file": str(file_path),
            "language": result["language"],
            "language_probability": round(result.get("language_probability", 0), 4),
            "transcript": transcript_text,
            "segments": segments,
            "intents": intents,
            "entities": entities,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Failed to transcribe {file_path}: {e}")
        return {
            "file": str(file_path),
            "status": "failed",
            "error": str(e),
            "processed_at": datetime.utcnow().isoformat() + "Z",
        }

def run_batch_pipeline(input_dir, output_dir, model_size="base"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    audio_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files. Loading model '{model_size}'...")
    model = whisper.load_model(model_size)

    results = []
    succeeded = 0
    failed = 0

    for audio_file in audio_files:
        logger.info(f"Processing: {audio_file.name}")
        result = transcribe_file(model, audio_file)
        results.append(result)

        if result["status"] == "success":
            succeeded += 1
            individual_output = output_path / f"{audio_file.stem}.json"
            with open(individual_output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"  → Wrote {individual_output}")
            logger.info(f"  → Language: {result['language']}, Intents: {list(result['intents'].keys())}")
        else:
            failed += 1

    summary = {
        "pipeline_run": datetime.utcnow().isoformat() + "Z",
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "model": model_size,
        "total_files": len(audio_files),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
    }

    summary_path = output_path / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPipeline complete: {succeeded} succeeded, {failed} failed")
    logger.info(f"Summary: {summary_path}")
    return summary

from gtts import gTTS

demo_dir = Path("call_recordings")
demo_dir.mkdir(exist_ok=True)

calls = [
    ("call_001_discovery", "Hi, this is Alex from TechCo. We are looking at your platform for our sales team of about fifty people. What does pricing look like? We are currently using Salesforce and HubSpot. Can you send over a one-pager? I would need to get my VP of Sales involved before we move forward."),
    ("call_002_objection", "Thanks for the demo. Honestly, our main concern is the implementation timeline. Our last vendor took six months and it was a disaster. Also, eighteen thousand per year feels steep compared to what we budgeted. Is there any flexibility? We are also evaluating Competitor Y next week."),
    ("call_003_closing", "Great, so we are aligned on the enterprise plan at twelve thousand per year. Next steps: I will send the security questionnaire by Wednesday, and you will introduce me to your CTO for final sign-off. Can we target a go-live date of March fifteenth?"),
]

for filename, script in calls:
    tts = gTTS(text=script, lang="en", slow=False)
    tts.save