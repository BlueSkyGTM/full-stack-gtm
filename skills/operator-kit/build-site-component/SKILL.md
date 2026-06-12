# /build-site-component

Generate site components, Helix implementation code, and student-facing UI logic.
Uses GLM-5 (strong coding + multi-step reasoning). Active from Stage 05.

## When to invoke
- Stage 05 Helix build is running
- A site component spec exists and needs implementation
- User says "build the [component]", "implement [feature] in the site"
- Helix gate logic, FSRS integration, or student state components

## Chain

### Step 1 — Read component spec + existing site patterns
```bash
SPEC="${1:-}"
SPEC_CONTENT=$(cat "$SPEC" 2>/dev/null || echo "No spec provided.")
# Sample existing site patterns for style reference
EXISTING=$(find site-new/ phases/ -name "*.js" -o -name "*.ts" -o -name "*.jsx" 2>/dev/null \
  | head -5 | xargs head -30 2>/dev/null)
HELIX_DESIGN=$(cat stages/00-d-helix-design/output/*.md 2>/dev/null | head -100)
```

### Step 2 — Call GLM-5
```bash
python3 - <<'PYEOF'
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

spec = sys.argv[1] if len(sys.argv) > 1 else ""
existing = sys.argv[2] if len(sys.argv) > 2 else ""
helix_design = sys.argv[3] if len(sys.argv) > 3 else ""

SYSTEM = """You are a frontend engineer building a GTM engineering curriculum site.
The site uses [framework from existing code]. You write clean, minimal, well-typed code.
Follow the patterns in the existing codebase. No unnecessary abstractions.
When building Helix (student memory/FSRS) components: keep state simple, surface review
prompts clearly, handle empty/loading/error states explicitly."""

USER = f"""Build this component:

Spec:
{spec[:2000]}

Existing site patterns (for style reference):
{existing[:2000]}

Helix design context:
{helix_design[:1000]}

Output the complete implementation. Include: component file, any required types/interfaces,
brief inline comments for non-obvious logic only."""

response = client.chat.completions.create(
    model="GLM-5",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
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
```bash
OUTPUT_DIR="stages/05-helix-build/output/components"
mkdir -p "$OUTPUT_DIR"
COMPONENT_NAME=$(basename "$SPEC" .md)
python3 [block above] "$SPEC_CONTENT" "$EXISTING" "$HELIX_DESIGN" > "$OUTPUT_DIR/${COMPONENT_NAME}.tsx"
echo "Written: $OUTPUT_DIR/${COMPONENT_NAME}.tsx"
```

### Step 4 — Review gate
Before merging any component to site-new/, invoke `/review` on the output file.
GLM-5 code output is good but always needs a review pass before it touches the live site.
