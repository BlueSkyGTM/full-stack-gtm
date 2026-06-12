# /write-lesson

Draft a lesson doc (`docs/en.md`) for a GTM engineering curriculum stage.
Uses GLM-5.1 (reasoning). Active from Stage 01.

## When to invoke
- Stage 01 is running and needs lesson outlines or full lesson drafts
- User says "write the lesson for [topic]" or "draft [stage] content"
- A stage CONTEXT.md calls for lesson generation

## Chain

### Step 1 — Read context
Load the stage's CONTEXT.md and any relevant variable-registry entries:
```bash
STAGE_CONTEXT=$(cat stages/01-gtm-skeleton/CONTEXT.md 2>/dev/null || echo "No context found")
VARIABLES=$(cat vault/variable-registry.md 2>/dev/null | head -60)
TOPIC="${1:-GTM engineering fundamentals}"
```
Adjust the stage path to whatever stage is active.

### Step 2 — Call GLM-5.1
```bash
python3 - <<'PYEOF'
import os, sys, json

sys.stdout.reconfigure(encoding='utf-8')

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

import sys
topic = sys.argv[1] if len(sys.argv) > 1 else "GTM engineering fundamentals"
stage_context = sys.argv[2] if len(sys.argv) > 2 else ""

SYSTEM = """You are a GTM engineering curriculum author. You write clear, practical lesson docs for technical GTM practitioners — people who sit between sales/marketing and engineering. Your lessons:
- Lead with a practical problem the reader actually faces
- Explain concepts through real GTM scenarios (CRM integrations, attribution, pipeline data)
- Include concrete examples with tool names (Salesforce, HubSpot, Segment, dbt, etc.)
- End with clear learning outcomes and a bridge to the next lesson
- Use plain English, active voice, no jargon without explanation
- Format: markdown, H2 sections, code blocks where relevant"""

USER = f"""Write a full lesson doc for: {topic}

Stage context:
{stage_context[:2000] if stage_context else "No additional context provided."}

Structure:
## Overview
## Why This Matters for GTM Engineers
## Core Concepts
## Hands-On Example
## Common Mistakes
## Learning Outcomes
## What's Next"""

response = client.chat.completions.create(
    model="GLM-5.1",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER},
    ],
    max_tokens=4000,
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
Save the draft to the active stage's output folder:
```bash
OUTPUT_DIR="stages/01-gtm-skeleton/output/lessons"
mkdir -p "$OUTPUT_DIR"
SLUG=$(echo "$TOPIC" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd 'a-z0-9-')
python3 [the block above] "$TOPIC" "$STAGE_CONTEXT" > "$OUTPUT_DIR/${SLUG}-draft.md"
echo "Written: $OUTPUT_DIR/${SLUG}-draft.md"
```

### Step 4 — Validate + report
- Confirm the file was written and is non-empty
- Check that all 6 H2 sections are present
- Report word count
- Flag to Claude Code if any section is under 50 words (likely truncated)

## Notes
- GLM-5.1 is a reasoning model: internal thinking tokens are not visible, output starts after reasoning completes. Normal.
- If output is empty, increase `max_tokens` — reasoning consumed the budget.
- Requires: `pip install openai` and `ZHIPUAI_API_KEY` + `ZAI_BASE_URL` in environment.
