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