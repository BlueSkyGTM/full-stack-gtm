# /build-site-component

Generate site components, Helix implementation code, and student-facing UI logic.
Uses GLM-5 (Lyra-code). Active from Stage 05.

## When to invoke
- Stage 05 Helix build is running
- A site component spec exists and needs implementation
- User says "build the [component]", "implement [feature] in the site"
- Helix gate logic, FSRS integration, or student state components

## Chain

### Step 0 — Pre-flight
```bash
python3 -c "import openai" 2>/dev/null || { echo "ERROR: pip install openai"; exit 1; }
[ -n "$ZHIPUAI_API_KEY" ] || { echo "ERROR: ZHIPUAI_API_KEY not set — check .env"; exit 1; }
[ -f "stages/00-c-agent-setup/output/agent-briefs/lyra-code-brief.md" ] || { echo "ERROR: lyra-code-brief.md missing — run Stage 00-c first"; exit 1; }
SPEC="${1:-}"
[ -f "$SPEC" ] || { echo "ERROR: spec file not found: $SPEC"; exit 1; }
```

### Step 1 — Read context (governed maze: extract only what GLM needs)
```bash
COMPONENT_NAME=$(basename "$SPEC" .md)

# Extract Lyra-code identity + architecture invariants + component writing rules + what NOT to do
LYRA_CODE_BRIEF=$(python3 -c "
import re
text = open('stages/00-c-agent-setup/output/agent-briefs/lyra-code-brief.md').read()
sections = ['## Identity', '## Architecture Invariants', '## Component Writing Rules', '## What NOT To Do']
out = []
for s in sections:
    m = re.search(re.escape(s) + r'(.*?)(?=\n## |\Z)', text, re.DOTALL)
    if m:
        out.append(s + m.group(1).rstrip())
print('\n\n'.join(out))
" 2>/dev/null | head -130)  # cap at ~480 tokens

# Helix integration rules (if this is a Helix component)
HELIX_RULES=$(python3 -c "
import re
text = open('stages/00-c-agent-setup/output/agent-briefs/lyra-code-brief.md').read()
m = re.search(r'## Helix Integration Rules.*?(.*?)(?=\n## |\Z)', text, re.DOTALL)
print('## Helix Integration Rules' + (m.group(1).rstrip() if m else ''))
" 2>/dev/null | head -30)

# Component spec (full — it's the task definition)
cp "$SPEC" /tmp/zai_comp_spec.txt

# Existing site patterns — capped snippet for style reference only
find site-new/ phases/ -name "*.tsx" -o -name "*.ts" 2>/dev/null \
  | head -3 | xargs head -25 2>/dev/null > /tmp/zai_comp_patterns.txt

echo "$LYRA_CODE_BRIEF" > /tmp/zai_lyracode_brief.txt
echo "$HELIX_RULES" > /tmp/zai_helix_rules.txt
```

### Step 2 — Call GLM-5 + write output
```bash
OUTPUT_DIR="stages/05-helix-build/output/components"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/${COMPONENT_NAME}.tsx"

python3 - /tmp/zai_lyracode_brief.txt /tmp/zai_helix_rules.txt /tmp/zai_comp_spec.txt /tmp/zai_comp_patterns.txt <<'PYEOF' > "$OUTPUT_FILE"
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

brief        = open(sys.argv[1]).read() if len(sys.argv) > 1 else ""
helix_rules  = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
spec         = open(sys.argv[3]).read() if len(sys.argv) > 3 else ""
patterns     = open(sys.argv[4]).read() if len(sys.argv) > 4 else ""

# System: Lyra-code identity + architecture invariants from brief (governed maze)
SYSTEM = f"""You are Lyra (code mode), a frontend engineer for a GTM engineering curriculum site.
{brief}"""

USER = f"""Build this component.

Spec (implement exactly as specified):
{spec[:2500]}

Helix integration rules (apply if this component touches student state or quiz flow):
{helix_rules[:600]}

Existing site patterns (match the style, do not copy):
{patterns[:800]}

Output the complete implementation. Types/interfaces inline. Comments only for
non-obvious invariants. No placeholder code — if something is unclear, write a
// TODO: [specific question] comment and implement the best-guess stub."""

response = client.chat.completions.create(
    model="GLM-5",
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

### Step 3 — Validate + report
```bash
BYTES=$(wc -c < "$OUTPUT_FILE" 2>/dev/null || echo 0)
if [ "$BYTES" -lt 100 ]; then
  echo "ERROR: output too small ($BYTES bytes) — likely API failure. Check $OUTPUT_FILE"
  rm -f /tmp/zai_lyracode_brief.txt /tmp/zai_helix_rules.txt /tmp/zai_comp_spec.txt /tmp/zai_comp_patterns.txt
  exit 1
fi
echo "Written: $OUTPUT_FILE ($BYTES bytes)"
rm -f /tmp/zai_lyracode_brief.txt /tmp/zai_helix_rules.txt /tmp/zai_comp_spec.txt /tmp/zai_comp_patterns.txt
```

### Step 4 — Review gate
```bash
echo "REQUIRED: invoke /review on $OUTPUT_FILE before merging to site-new/"
echo "GLM-5 code output is good but always needs a review pass before touching the live site."
```

## Notes
- Lyra-code brief loaded at runtime (~480 tokens): identity, architecture invariants, component rules, what NOT to do
- Helix integration rules loaded separately (only relevant for Stage 05 components)
- Governed maze: GLM receives brief extract + spec + thin style snippet. Not the full codebase.
- Output goes to stages/05-helix-build/output/components/ — Claude Code reviews before moving to site-new/
- /review is mandatory before any component touches the live site
