## Ship It

Deploying multilingual processing as a preprocessing stage in an enrichment pipeline requires three production decisions.

**Translate-then-classify versus classify-in-original-language.** Classify in the original language when entity boundary preservation matters (NER, relation extraction, firmographic field extraction). Translate first when the downstream task operates on full-text semantics (sentiment analysis, topic classification, intent detection) and when your classification model is monolingual — an English-only classifier fed translated text will often outperform a multilingual classifier fed original-language text, because the English classifier had more training data. The tradeoff is latency: translation adds 50-200ms per record depending on model size, and it introduces a failure mode where translation errors propagate into classification errors invisibly. For high-throughput enrichment pipelines processing thousands of prospects, classify-in-original-language with a multilingual model is the default; switch to translate-then-classify only when per-language accuracy auditing shows the multilingual model is failing on a specific language.

**Language detection on short text.** Company names and job titles are 2-6 words. Language detection models like `langdetect` or `fasttext-langdetect` are trained on document-length text and produce unreliable predictions on short inputs — a job title like "Director" could be English, French, or Spanish. The production fix is to run language detection on the longest available text field (company description, not company name) and propagate that language label to shorter fields from the same record. If only short text is available, set a confidence floor: if `langdetect` returns confidence below 0.7, flag the record as `language_uncertain` and default to English processing with a metadata flag indicating low confidence.

**Per-language confidence thresholds.** XLM-R's NER performance varies by language based on how much of that language appeared in the pretraining corpus. English, French, German, and Spanish have strong cross-lingual transfer. Chinese, Japanese, and Arabic have moderate transfer. Low-resource languages like Yoruba or Maithili have weak transfer. Set confidence thresholds inversely proportional to expected model quality: require 0.85 confidence for English entity extraction, but accept 0.70 for Japanese, because the model's baseline confidence is lower even on correct predictions. Calibrate these thresholds by running the model on a held-out set of manually labeled examples per language and measuring precision-recall curves — without this calibration, you will either over-trust low-quality extractions or reject too many valid ones.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from langdetect import detect_langs
import unicodedata
import json

ner_pipe = pipeline(
    "ner",
    model="xlm-roberta-large-finetuned-conll03-english",
    aggregation_strategy="simple"
)

LANGUAGE_CONFIDENCE_FLOORS = {
    "en": 0.85,
    "de": 0.78,
    "fr": 0.78,
    "es": 0.78,
    "pt": 0.78,
    "ja": 0.65,
    "zh": 0.65,
    "ar": 0.65,
    "default": 0.70
}

def detect_language(text):
    if len(text.split()) < 4:
        return None, 0.0, "short_text"
    try:
        detections = detect_langs(text)
        top = detections[0]
        return top.lang, top.prob, None
    except Exception:
        return None, 0.0, "detection_failed"

def enrich_prospect(company_name, company_description):
    normalized = unicodedata.normalize("NFC", company_description or company_name)
    
    lang, lang_conf, warning = detect_language(normalized)
    
    if lang is None or lang_conf < 0.70:
        lang = "en"
        lang_conf = lang_conf if lang_conf else 0.0
        warning = warning or "low_confidence_default_en"
    
    entities = ner_pipe(normalized)
    threshold = LANGUAGE_CONFIDENCE_FLOORS.get(lang, LANGUAGE_CONFIDENCE_FLOORS["default"])
    
    org_entities = [
        {
            "text": ent["word"],
            "confidence": round(ent["score"], 3),
            "accepted": ent["score"] >= threshold
        }
        for ent in entities
        if ent["entity_group"] == "ORG"
    ]
    
    return {
        "company_name": company_name,
        "detected_language": lang,
        "language_confidence": round(lang_conf, 3),
        "language_warning": warning,
        "ner_threshold_used": threshold,
        "organizations": org_entities,
        "processing_path": "classify_original_language"
    }

test_records = [
    ("Siemens AG", "Siemens AG ist ein deutsches multinationales Konglomerat mit Hauptsitz in München."),
    ("Toyota", "Toyota Motor Corporation manufactures automobiles and commercial vehicles."),
    ("LVMH", "LVMH est une entreprise française spécialisée dans les produits de luxe et les spiritueux."),
    ("ShortCo", "Tech company")
]

for name, desc in test_records:
    result = enrich_prospect(name, desc)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
```

The `language_confidence` field is the key output for downstream pipeline logic. Records with `language_warning` set to `low_confidence_default_en` should be routed to manual review or flagged for re-processing when better text data becomes available. The `ner_threshold_used` field makes the per-language threshold auditable — if a stakeholder asks why an entity was accepted at 0.68 confidence, the record shows it was processed under Japanese thresholds.