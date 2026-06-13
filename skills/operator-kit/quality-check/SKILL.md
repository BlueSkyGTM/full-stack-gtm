# /quality-check

Curriculum audit: accuracy, gaps, source alignment, coverage.
Uses GLM-4.7 (Hypatia). Active at Stage 09, callable earlier for spot checks.

## When to invoke
- Stage 09 quality pass is running
- A lesson or exercise needs accuracy review
- User says "check this", "audit", "does this match the source", "what's missing"
- After `/write-lesson` or `/write-exercise` produces a draft

## Chain

### Step 0 — Pre-flight
```bash
python3 -c "import openai" 2>/dev/null || { echo "ERROR: pip install openai"; exit 1; }
[ -n "$ZHIPUAI_API_KEY" ] || { echo "ERROR: ZHIPUAI_API_KEY not set — check .env"; exit 1; }
[ -f "stages/00-c-agent-setup/output/agent-briefs/hypatia-brief.md" ] || { echo "ERROR: hypatia-brief.md missing — run Stage 00-c first"; exit 1; }
```

### Step 1 — Read context (governed maze: extract only what GLM needs)
```bash
TARGET="${1:-stages/02-lesson-injection/output}"
[ -e "$TARGET" ] || { echo "ERROR: target not found: $TARGET"; exit 1; }

# Extract Hypatia identity + audit checklist + verdict format + what NOT to do
HYPATIA_BRIEF=$(python3 -c "
import re
text = open('stages/00-c-agent-setup/output/agent-briefs/hypatia-brief.md').read()
sections = ['## Identity', '## Audit Checklist', '## Verdict Format', '## What NOT To Do']
out = []
for s in sections:
    m = re.search(re.escape(s) + r'(.*?)(?=\n## |\Z)', text, re.DOTALL)
    if m:
        out.append(s + m.group(1).rstrip())
print('\n\n'.join(out))
" 2>/dev/null | head -120)  # cap at ~400 tokens

# Extract format specs (what a correct lesson/quiz looks like — Hypatia checks against these)
FORMAT_RULES=$(python3 -c "
import re
spec = open('stages/00-a-curriculum-archaeology/output/lesson-format-spec.md').read()
m = re.search(r'## Rules(.*?)(?=\n## |\Z)', spec, re.DOTALL)
print('## Lesson Rules' + (m.group(1).rstrip() if m else ''))
" 2>/dev/null | head -30)

# Collect target content (first 5 markdown files, capped)
find "$TARGET" -name "*.md" 2>/dev/null | head -5 | xargs cat 2>/dev/null | head -300 > /tmp/zai_audit_target.txt
head -80 stages/00-b-gtm-content-mapping/output/source-citations.md 2>/dev/null > /tmp/zai_audit_citations.txt
echo "$HYPATIA_BRIEF" > /tmp/zai_hypatia_brief.txt
echo "$FORMAT_RULES" > /tmp/zai_format_rules.txt
```

### Step 2 — Call GLM-4.7 + write audit report
```bash
OUTPUT_DIR="stages/09-quality-pass/output"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/audit-${TIMESTAMP}.md"

python3 - /tmp/zai_hypatia_brief.txt /tmp/zai_format_rules.txt /tmp/zai_audit_target.txt /tmp/zai_audit_citations.txt <<'PYEOF' > "$OUTPUT_FILE"
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

brief        = open(sys.argv[1]).read() if len(sys.argv) > 1 else ""
format_rules = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
target       = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
citations    = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""

# System: Hypatia identity + core rules from brief (governed maze)
SYSTEM = f"""You are Hypatia, a GTM curriculum quality auditor.
{brief}"""

USER = f"""Audit this curriculum content. Apply the lesson format rules and source fidelity rules below.

Format rules (what correct looks like):
{format_rules}

Reference sources (check claims against these):
{citations[:1500]}

Content to audit:
{target[:3500]}

Use the Verdict Format from your brief exactly. Every criterion gets PASS/WARN/FAIL.
End with Overall Verdict: SHIP / REVISE / BLOCK."""

response = client.chat.completions.create(
    model="GLM-4.7",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER},
    ],
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

### Step 3 — Validate + escalate
```bash
BYTES=$(wc -c < "$OUTPUT_FILE" 2>/dev/null || echo 0)
if [ "$BYTES" -lt 100 ]; then
  echo "ERROR: audit output too small ($BYTES bytes) — likely API failure"
  rm -f /tmp/zai_audit_target.txt /tmp/zai_audit_citations.txt /tmp/zai_hypatia_brief.txt /tmp/zai_format_rules.txt
  exit 1
fi
echo "Audit written: $OUTPUT_FILE"

# Escalate BLOCK verdicts immediately
if grep -q "Overall Verdict.*BLOCK\|BLOCK" "$OUTPUT_FILE" 2>/dev/null; then
  echo ""
  echo "!!! BLOCK VERDICT — halting pipeline. Review $OUTPUT_FILE before proceeding to Stage 10."
  rm -f /tmp/zai_audit_target.txt /tmp/zai_audit_citations.txt /tmp/zai_hypatia_brief.txt /tmp/zai_format_rules.txt
  exit 1
fi

rm -f /tmp/zai_audit_target.txt /tmp/zai_audit_citations.txt /tmp/zai_hypatia_brief.txt /tmp/zai_format_rules.txt
echo "Verdict: $(grep -i 'Overall Verdict' "$OUTPUT_FILE" | head -1)"
```

## Notes
- Hypatia's brief is loaded at runtime (~400 tokens): identity, audit checklist, verdict format, what NOT to do
- Governed maze: GLM receives the brief extract + lesson format rules + target content. Not the whole repo.
- BLOCK verdict exits with code 1 — pipeline halts. REVISE continues but flags the file.
- For spot-check on a single file, pass the file path directly: `/quality-check phases/01-math/01-linear-algebra/docs/en.md`
