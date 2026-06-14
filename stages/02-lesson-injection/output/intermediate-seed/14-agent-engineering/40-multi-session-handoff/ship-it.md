## Ship It

Production handoff requires four components working together. First, a **deterministic session key** — typically `account_id` or `prospect_id` — that maps to exactly one capsule per active workflow. Ambiguous keys (e.g., email address when a prospect changes companies) cause cross-contamination. Second, a **storage layer** with TTL and conflict resolution: capsules for active opportunities should persist; capsules for closed-lost accounts should expire. Third, a **rehydration validator** that rejects corrupted or schema-stale capsules rather than silently loading garbage. Fourth, a **fallback path** — when the capsule is missing or invalid, the session must degrade gracefully to a fresh start, not crash.

```python
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

CURRENT_VERSION = 3
DEFAULT_TTL_SECONDS = 7 * 24 * 3600

@dataclass
class Capsule:
    version: int
    session_key: str
    entities: dict
    decisions: list
    open_questions: list
    confidence_scores: dict
    created_at: float
    ttl_seconds: int

def write_capsule(session_key, entities, decisions, open_questions,
                  confidence_scores=None, ttl_seconds=DEFAULT_TTL_SECONDS):
    capsule = Capsule(
        version=CURRENT_VERSION,
        session_key=session_key,
        entities=entities,
        decisions=decisions,
        open_questions=open_questions,
        confidence_scores=confidence_scores or {},
        created_at=time.time(),
        ttl_seconds=ttl_seconds
    )
    path = Path(f"/tmp/capsules/{session_key}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(capsule), indent=2))
    return capsule

def validate_capsule(raw):
    errors = []
    if not isinstance(raw, dict):
        return ["Capsule is not a dict"]
    if raw.get("version") != CURRENT_VERSION:
        errors.append(f"Version mismatch: got {raw.get('version')}, expected {CURRENT_VERSION}")
    for field in ["session_key", "entities", "decisions", "created_at"]:
        if field not in raw:
            errors.append(f"Missing required field: {field}")
    if raw.get("created_at") and time.time() - raw["created_at"] > raw.get("ttl_seconds", DEFAULT_TTL_SECONDS):
        errors.append("Capsule expired (TTL exceeded)")
    return errors

def read_capsule(session_key):
    path = Path(f"/tmp/capsules/{session_key}.json")
    if not path.exists():
        print(f"[HANDOFF] No capsule for {session_key}. Starting fresh.")
        return None
    raw = json.loads(path.read_text())
    errors = validate_capsule(raw)
    if errors:
        print(f"[HANDOFF] Capsule rejected for {session_key}:")
        for e in errors:
            print(f"  - {e}")
        print("[HANDOFF] Degrading to fresh session.")
        return None
    return Capsule(**raw)

def inject_into_prompt(capsule):
    if capsule is None:
        return "You are starting a fresh session with no prior context."
    return f"""Prior session context loaded.
Account: {capsule.entities.get('account', 'unknown')}
Key decisions: {'; '.join(capsule.decisions)}
Open questions: {'; '.join(capsule.open_questions)}
Proceed from here."""

print("=== WRITE: Session A serializes research findings ===")
capsule = write_capsule(
    "acme_corp",
    entities={"account": "Acme Corp", "dm": "Jane Doe", "competitor": "Initech"},
    decisions=["Target Jane Doe", "Use competitor displacement angle"],
    open_questions=["Jane's LinkedIn activity?", "Q1 budget status?"],
    confidence_scores={"account": 0.95, "dm": 0.80, "competitor": 0.70}
)
print(f"Written: {capsule.session_key} (v{capsule.version})")

print("\n=== READ: Session B rehydrates for outreach ===")
loaded = read_capsule("acme_corp")
prompt = inject_into_prompt(loaded)
print(prompt)

print("\n=== EDGE CASE: Missing capsule ===")
loaded_missing = read_capsule("nonexistent_account")
print(inject_into_prompt(loaded_missing))

print("\n=== EDGE CASE: Corrupted capsule ===")
corrupt_path = Path("/tmp/capsules/corrupt_co.json")
corrupt_path.parent.mkdir(parents=True, exist_ok=True)
corrupt_path.write_text('{"version": 1, "broken": true}')
loaded_corrupt = read_capsule("corrupt_co")
print(inject_into_prompt(loaded_corrupt))

print("\n=== EDGE CASE: Expired capsule ===")
expired = write_capsule(
    "old_lead",
    entities={"account": "Old Co"},
    decisions=["Decided to pass"],
    open_questions=[],
    ttl_seconds=0
)
time.sleep(0.1)
loaded_expired = read_capsule("old_lead")
print(inject_into_prompt(loaded_expired))

print("\n=== PRODUCTION CHECKLIST ===")
print("1. Session key deterministic: account_id or prospect_id")
print("2. TTL configured per use case (active vs closed-lost)")
print("3. Validator rejects version mismatch and missing fields")
print("4. Fallback returns fresh-session prompt, never crashes")
print("5. All four edge cases above passed without exception")
```