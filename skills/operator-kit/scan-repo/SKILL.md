# /scan-repo

Fast read-only codebase traversal. Maps structure, finds patterns, surfaces gaps.
Uses GLM-5-Turbo (Echo — fast, low cost). Active from Stage 01+.

Read-only: this skill never writes files. Output goes directly to Claude Code
as context for next steps.

## When to invoke
- Need to understand what exists before building
- Looking for a file, pattern, or symbol across the repo
- Stage entry check: "what's already here for [topic]?"
- User says "scan", "find", "what do we have for", "check if X exists"

## Chain

### Step 0 — Pre-flight
```bash
python3 -c "import openai" 2>/dev/null || { echo "ERROR: pip install openai"; exit 1; }
[ -n "$ZHIPUAI_API_KEY" ] || { echo "ERROR: ZHIPUAI_API_KEY not set — check .env"; exit 1; }
[ -f "stages/00-c-agent-setup/output/agent-briefs/echo-brief.md" ] || { echo "ERROR: echo-brief.md missing — run Stage 00-c first"; exit 1; }
```

### Step 1 — Read context (governed maze: extract only what GLM needs)
```bash
QUERY="${1:-summarize the repo structure}"

# Extract Echo identity + primary role + format constraints + what NOT to do
ECHO_BRIEF=$(python3 -c "
import re
text = open('stages/00-c-agent-setup/output/agent-briefs/echo-brief.md').read()
sections = ['## Identity', '## Primary Role', '## Format Constraints', '## What NOT To Do']
out = []
for s in sections:
    m = re.search(re.escape(s) + r'(.*?)(?=\n## |\Z)', text, re.DOTALL)
    if m:
        out.append(s + m.group(1).rstrip())
print('\n\n'.join(out))
" 2>/dev/null | head -90)  # cap at ~280 tokens

# File tree (gitignored paths excluded)
TREE=$(find . \
  -not -path './.git/*' -not -path './node_modules/*' \
  -not -path './.claude/*' -not -path './graphify-out/*' \
  -not -path './__pycache__/*' -not -path './.gbrain/*' \
  | sort | head -200)

# Relevant content: files whose paths or content match the query
RELEVANT=$(grep -rl "$QUERY" stages/ vault/ references/ skills/ shared/ 2>/dev/null | head -8)
CONTENT=""
for f in $RELEVANT; do
  CONTENT="$CONTENT\n\n--- $f ---\n$(head -35 "$f" 2>/dev/null)"
done

printf '%s' "$ECHO_BRIEF" > /tmp/zai_echo_brief.txt
printf '%s' "$TREE" > /tmp/zai_scan_tree.txt
printf '%b' "$CONTENT" > /tmp/zai_scan_content.txt
```

### Step 2 — Call GLM-5-Turbo (output to Claude Code, not a file)
```bash
python3 - "$QUERY" /tmp/zai_echo_brief.txt /tmp/zai_scan_tree.txt /tmp/zai_scan_content.txt <<'PYEOF'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

query   = sys.argv[1] if len(sys.argv) > 1 else "summarize the repo structure"
brief   = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
tree    = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
content = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""

# System: Echo identity + format rules from brief (governed maze)
SYSTEM = f"""You are Echo, a codebase navigator for a GTM engineering curriculum project.
{brief}"""

USER = f"""Query: {query}

File tree (first 200 entries):
{tree[:2500]}

Relevant file content:
{content[:2500]}

Answer the query. Be specific — include exact file paths. Flag any gaps or
missing files that are relevant to the query. Keep output concise."""

response = client.chat.completions.create(
    model="GLM-5-Turbo",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER},
    ],
    max_tokens=1000,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
PYEOF

rm -f /tmp/zai_echo_brief.txt /tmp/zai_scan_tree.txt /tmp/zai_scan_content.txt
```

## Notes
- Echo's brief is loaded at runtime (~280 tokens): identity, primary role, format constraints, what NOT to do
- Governed maze: GLM receives brief extract + filtered tree + matching content snippets. Not the full repo.
- No file output — result goes to stdout (Claude Code reads it and decides next steps)
- For deep content analysis use `/quality-check` instead; for symbol lookup use `gbrain code-def`
