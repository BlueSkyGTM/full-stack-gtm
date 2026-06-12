# /write-exercise

Generate exercise specs for a GTM engineering curriculum lesson.
Uses GLM-5.1 (reasoning). Active from Stage 03.

## When to invoke
- Stage 03 exercise design is running
- A lesson draft exists and needs hands-on practice tasks
- User says "write exercises for [lesson]", "build the exercise for [topic]"

## Chain

### Step 1 — Read lesson content
```bash
LESSON="${1:-}"
LESSON_CONTENT=$(cat "$LESSON" 2>/dev/null || echo "No lesson file specified.")
TOPIC=$(head -3 "$LESSON" 2>/dev/null | grep "^#" | sed 's/^#*//' | xargs)
```

### Step 2 — Call GLM-5.1
```bash
python3 - <<'PYEOF'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

topic = sys.argv[1] if len(sys.argv) > 1 else ""
lesson_content = sys.argv[2] if len(sys.argv) > 2 else ""

SYSTEM = """You design practical exercises for GTM engineering students. Each exercise:
- Ties directly to a lesson's learning outcomes (never generic)
- Uses realistic GTM scenarios: data pipelines, CRM config, attribution logic, reporting
- Has a clear setup, task, expected output, and success criteria
- Includes a worked example and common failure modes
- Is completable in 20-45 minutes"""

USER = f"""Design 2-3 exercises for this lesson:

Topic: {topic}

Lesson content:
{lesson_content[:3000]}

For each exercise provide:
## Exercise [N]: [Name]
**Time:** [estimate]
**Scenario:** [realistic GTM context]
**Setup:** [what the student has before starting]
**Task:** [what they must do]
**Expected Output:** [what correct completion looks like]
**Success Criteria:** [how to verify]
**Worked Example:** [abbreviated solution path]
**Common Mistakes:** [2-3 failure modes]"""

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
OUTPUT_DIR="stages/03-exercise-design/output"
mkdir -p "$OUTPUT_DIR"
SLUG=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-')
python3 [block above] "$TOPIC" "$LESSON_CONTENT" > "$OUTPUT_DIR/${SLUG}-exercises.md"
echo "Written: $OUTPUT_DIR/${SLUG}-exercises.md"
```
