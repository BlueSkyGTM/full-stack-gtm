# /find-citations

Gap-fill research: surfaces citation pointers and search queries for undercited claims.
Uses GLM-4.5-Air (Newton — lightweight, fast). Active from Stage 01+.

Output is citation pointers and search queries, NOT verified URLs. GLM cannot
browse; URLs from GLM are hallucinations. Use output to guide manual lookup.

## When to invoke
- `/quality-check` flags unsupported claims or coverage gaps
- A lesson references a concept without a source
- Stage 00-b GTM content mapping needs additional citations
- User says "find a source for", "what backs this up", "citations for [topic]"

## Chain

### Step 0 — Pre-flight
```bash
python3 -c "import openai" 2>/dev/null || { echo "ERROR: pip install openai"; exit 1; }
[ -n "$ZHIPUAI_API_KEY" ] || { echo "ERROR: ZHIPUAI_API_KEY not set — check .env"; exit 1; }
[ -f "stages/00-c-agent-setup/output/agent-briefs/newton-brief.md" ] || { echo "ERROR: newton-brief.md missing — run Stage 00-c first"; exit 1; }
```

### Step 1 — Read context (governed maze: extract only what GLM needs)
```bash
TOPIC="${1:-GTM engineering fundamentals}"

# Extract Newton identity + scope + what Newton produces + what NOT to do
NEWTON_BRIEF=$(python3 -c "
import re
text = open('stages/00-c-agent-setup/output/agent-briefs/newton-brief.md').read()
sections = ['## Identity', '## Scope', '## What Newton Produces', '## What NOT To Do']
out = []
for s in sections:
    m = re.search(re.escape(s) + r'(.*?)(?=\n## |\Z)', text, re.DOTALL)
    if m:
        out.append(s + m.group(1).rstrip())
print('\n\n'.join(out))
" 2>/dev/null | head -100)  # cap at ~300 tokens

# Existing citations for this topic area (avoid duplicates)
grep -i "$TOPIC" stages/00-b-gtm-content-mapping/output/source-citations.md 2>/dev/null | head -20 > /tmp/zai_cite_existing.txt

# Gaps flagged by quality audit (if any recent audit exists)
grep -A2 "Coverage Gaps\|WARN\|missing" stages/09-quality-pass/output/*.md 2>/dev/null | head -30 > /tmp/zai_cite_gaps.txt

echo "$NEWTON_BRIEF" > /tmp/zai_newton_brief.txt
```

### Step 2 — Call GLM-4.5-Air + append to citations file
```bash
CITATIONS_FILE="stages/00-b-gtm-content-mapping/output/source-citations.md"

python3 - "$TOPIC" /tmp/zai_newton_brief.txt /tmp/zai_cite_existing.txt /tmp/zai_cite_gaps.txt <<'PYEOF' >> "$CITATIONS_FILE"
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

topic    = sys.argv[1] if len(sys.argv) > 1 else ""
brief    = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
existing = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
gaps     = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""

# System: Newton identity + rules from brief (governed maze)
SYSTEM = f"""You are Newton, a GTM curriculum research assistant.
{brief}"""

USER = f"""Surface 5-10 citation pointers for: {topic}

Existing citations already in the file (do not duplicate):
{existing[:800] if existing.strip() else "None yet."}

Gaps flagged by quality audit:
{gaps[:400] if gaps.strip() else "None specified."}

For each pointer:
- **[Publication / Source type]** — [specific claim to verify] | Search: "[search terms]"

Do NOT invent URLs. Surface where to look and what exact claim to search for.
Mark with datestamp: <!-- newton {topic} -->"""

response = client.chat.completions.create(
    model="GLM-4.5-Air",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER},
    ],
    max_tokens=1500,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
PYEOF
```

### Step 3 — Validate + report
```bash
BYTES=$(wc -c < "$CITATIONS_FILE" 2>/dev/null || echo 0)
if [ "$BYTES" -lt 50 ]; then
  echo "ERROR: citations file empty — likely API failure"
  rm -f /tmp/zai_newton_brief.txt /tmp/zai_cite_existing.txt /tmp/zai_cite_gaps.txt
  exit 1
fi
echo "Appended citation pointers to: $CITATIONS_FILE"
echo "NOTE: these are search pointers, not verified URLs — look up the actual sources manually."
rm -f /tmp/zai_newton_brief.txt /tmp/zai_cite_existing.txt /tmp/zai_cite_gaps.txt
```

## Notes
- Newton's brief is loaded at runtime (~300 tokens): identity, scope, what to produce, what NOT to do
- Output appends to source-citations.md — never overwrites existing citations
- Governed maze: GLM receives brief extract + existing citations + gap flags. Not the full repo.
- URLs from GLM are hallucinations — this skill returns WHERE to look and WHAT to search, not links
