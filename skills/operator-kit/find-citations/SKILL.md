# /find-citations

Gap-fill research and citation finding. Surfaces source material for undercited claims.
Uses GLM-4.5-Air (lightweight, fast). Active from Stage 01+.

## When to invoke
- `/quality-check` flags unsupported claims or coverage gaps
- A lesson references a concept without a source
- Stage 00-b GTM content mapping needs additional citations
- User says "find a source for", "what backs this up", "citations for [topic]"

## Chain

### Step 1 — Identify gaps
```bash
TOPIC="${1:-GTM engineering fundamentals}"
EXISTING_CITATIONS=$(cat stages/00-b-gtm-content-mapping/output/source-citations.md 2>/dev/null | grep -i "$TOPIC" | head -20)
AUDIT_GAPS=$(cat stages/09-quality-pass/output/*.md 2>/dev/null | grep -A2 "Coverage Gaps" | head -30)
```

### Step 2 — Call GLM-4.5-Air
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
existing = sys.argv[2] if len(sys.argv) > 2 else ""
gaps = sys.argv[3] if len(sys.argv) > 3 else ""

SYSTEM = """You are a research assistant for a GTM engineering curriculum.
You find and summarize credible sources for GTM engineering concepts.
Sources: industry blogs (a16z, Sequoia, FirstRound), vendor docs (Salesforce, HubSpot,
Segment, dbt, Fivetran), analyst reports, practitioner newsletters (RevOps Co-op,
GTM Alliance), and technical documentation. Always include: source name, type, URL if known, key claim."""

USER = f"""Find citations for: {topic}

Existing citations we already have:
{existing[:1000] if existing else "None yet."}

Gaps flagged by quality audit:
{gaps[:500] if gaps else "None specified."}

Return 5-10 citations. Format each as:
- **[Source Name]** ([type]) — [key claim relevant to topic]. URL: [if known]"""

response = client.chat.completions.create(
    model="GLM-4.5-Air",
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
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

### Step 3 — Append to citations file
```bash
CITATIONS_FILE="stages/00-b-gtm-content-mapping/output/source-citations.md"
mkdir -p "$(dirname $CITATIONS_FILE)"
echo "\n\n## Citations for: $TOPIC (added $(date +%Y-%m-%d))" >> "$CITATIONS_FILE"
python3 [block above] "$TOPIC" "$EXISTING_CITATIONS" "$AUDIT_GAPS" >> "$CITATIONS_FILE"
echo "Appended to: $CITATIONS_FILE"
```
