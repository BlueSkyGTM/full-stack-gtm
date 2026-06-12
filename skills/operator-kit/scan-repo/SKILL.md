# /scan-repo

Fast read-only codebase traversal. Maps structure, finds patterns, surfaces gaps.
Uses GLM-4.7-Flash (fast, low cost). Active from Stage 01+.

## When to invoke
- Need to understand what exists before building
- Looking for a file, pattern, or symbol across the repo
- Stage entry check: "what's already here for [topic]?"
- User says "scan", "find", "what do we have for", "check if X exists"

## Chain

### Step 1 — Collect file tree + target content
```bash
QUERY="${1:-summarize the repo structure}"
TREE=$(find . -not -path './.git/*' -not -path './node_modules/*' -not -path './.claude/*' \
  -not -path './graphify-out/*' -not -path './__pycache__/*' \
  | sort | head -200)
# Pull content of files that match the query topic
RELEVANT=$(grep -rl "$QUERY" stages/ vault/ references/ skills/ 2>/dev/null | head -10)
CONTENT=""
for f in $RELEVANT; do
  CONTENT="$CONTENT\n\n--- $f ---\n$(head -40 $f)"
done
```

### Step 2 — Call GLM-4.7-Flash
```bash
python3 - <<'PYEOF'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

query = sys.argv[1] if len(sys.argv) > 1 else "summarize the repo structure"
tree  = sys.argv[2] if len(sys.argv) > 2 else ""
content = sys.argv[3] if len(sys.argv) > 3 else ""

SYSTEM = """You are a codebase navigator for a GTM engineering curriculum project.
You read file trees and content snippets and give concise, actionable summaries.
Report: what exists, what's missing, what needs attention. Be specific with file paths."""

USER = f"""Query: {query}

File tree (first 200 entries):
{tree[:3000]}

Relevant file content:
{content[:3000]}

Answer the query. Be specific. Include file paths. Flag any gaps relevant to the query."""

response = client.chat.completions.create(
    model="GLM-4.7-Flash",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    max_tokens=1000,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
PYEOF
```

### Step 3 — Report
Output goes directly to Claude Code (no file write — scan is read-only).
Claude Code uses the result to decide next steps.

## Notes
- GLM-4.7-Flash: standard model (not reasoning), fast responses, low token cost.
- Read-only: this skill never writes files.
- For deep code analysis use `/quality-check` instead.
