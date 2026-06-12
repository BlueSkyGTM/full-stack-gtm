# /quality-check

Curriculum audit: accuracy, gaps, source alignment, coverage.
Uses GLM-4.7. Active at Stage 09, callable earlier for spot checks.

## When to invoke
- Stage 09 quality pass is running
- A lesson or exercise needs accuracy review
- User says "check this", "audit", "does this match the source", "what's missing"
- After `/write-lesson` or `/write-exercise` produces a draft

## Chain

### Step 1 — Load target + sources
```bash
TARGET="${1:-stages/02-lesson-injection/output}"
CITATIONS=$(cat stages/00-b-gtm-content-mapping/output/source-citations.md 2>/dev/null | head -100)
TARGET_CONTENT=$(find "$TARGET" -name "*.md" | head -5 | xargs cat 2>/dev/null | head -300)
```

### Step 2 — Call GLM-4.7
```bash
python3 - <<'PYEOF'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

target_content = sys.argv[1] if len(sys.argv) > 1 else ""
citations = sys.argv[2] if len(sys.argv) > 2 else ""

SYSTEM = """You are a curriculum quality auditor for a GTM engineering course.
You check lesson content for: factual accuracy, alignment with cited sources,
coverage gaps, unclear explanations, and unsupported claims.
Output a structured audit report: PASS / WARN / FAIL per criterion, with specific line references."""

USER = f"""Audit this curriculum content:

{target_content[:4000]}

Reference sources:
{citations[:2000]}

Report format:
## Accuracy — [PASS/WARN/FAIL]
[findings]
## Source Alignment — [PASS/WARN/FAIL]
[findings]
## Coverage Gaps — [list any]
## Unclear Explanations — [list any]
## Overall Verdict — [SHIP / REVISE / BLOCK]"""

response = client.chat.completions.create(
    model="GLM-4.7",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    max_tokens=2000,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
PYEOF
```

### Step 3 — Write audit report
```bash
OUTPUT_DIR="stages/09-quality-pass/output"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
python3 [block above] "$TARGET_CONTENT" "$CITATIONS" > "$OUTPUT_DIR/audit-${TIMESTAMP}.md"
echo "Audit written: $OUTPUT_DIR/audit-${TIMESTAMP}.md"
```

### Step 4 — Escalate if BLOCK
If the audit report contains `## Overall Verdict — BLOCK`, surface it to Claude Code
immediately and halt the stage pipeline. Do not proceed to Stage 10.
