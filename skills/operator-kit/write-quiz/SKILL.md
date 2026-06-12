# /write-quiz

Generate FSRS-ready quiz banks for a lesson. Questions grounded in lesson content only.
Uses GLM-5.1 (reasoning). Active from Stage 04.

## When to invoke
- Stage 04 quiz/recall design is running
- A lesson doc exists and needs quiz questions
- User says "write quiz for [lesson]", "generate questions for [topic]"
- Never run before the lesson's `docs/en.md` objectives are confirmed

## Chain

### Step 1 — Read lesson + objectives
```bash
LESSON_DOC="${1:-}"
CONTENT=$(cat "$LESSON_DOC" 2>/dev/null || echo "No lesson specified.")
# Extract learning outcomes section
OBJECTIVES=$(echo "$CONTENT" | awk '/## Learning Outcomes/,/^## /' | head -20)
```

### Step 2 — Call GLM-5.1
```bash
python3 - <<'PYEOF'
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

lesson_content = sys.argv[1] if len(sys.argv) > 1 else ""
objectives = sys.argv[2] if len(sys.argv) > 2 else ""

SYSTEM = """You write quiz questions for a GTM engineering curriculum.
Rules:
- Every question must be answerable from the lesson content — no outside knowledge required
- Questions test understanding, not memorization of wording
- 4 answer choices per question: 1 correct, 3 plausible distractors
- Include a brief explanation for the correct answer
- Distribute across: recall (30%), application (50%), analysis (20%)
Output as JSON array."""

USER = f"""Write 5 quiz questions for this lesson.

Learning objectives:
{objectives[:500]}

Lesson content:
{lesson_content[:4000]}

Output JSON:
[
  {{
    "id": "q1",
    "type": "recall|application|analysis",
    "question": "...",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct": "A",
    "explanation": "..."
  }}
]"""

response = client.chat.completions.create(
    model="GLM-5.1",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    max_tokens=3000,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
PYEOF
```

### Step 3 — Write output
```bash
OUTPUT_DIR="stages/04-quiz-recall/output"
mkdir -p "$OUTPUT_DIR"
LESSON_NAME=$(basename "$LESSON_DOC" .md)
python3 [block above] "$CONTENT" "$OBJECTIVES" > "$OUTPUT_DIR/${LESSON_NAME}-quiz.json"
echo "Written: $OUTPUT_DIR/${LESSON_NAME}-quiz.json"
```

### Step 4 — Validate
Run `python3 scripts/audit_lessons.py` after writing if available.
If the JSON is malformed, surface the raw output to Claude Code for repair.
