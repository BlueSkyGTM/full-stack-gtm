<!-- /autoplan restore point: /c/Users/raymo/.gstack/projects/BlueSkyGTM-blueskygtm-engineering/master-autoplan-restore-20260612-055742.md -->
# Operator-Kit Architecture Plan: Agentic GLM Skills

## Architecture (current — post Railway proxy scrap)

Each operator-kit entry is a **gstack agentic skill** — a SKILL.md with embedded
chain logic that orchestrates a full workflow: read context → call Z.ai GLM →
write output → validate. No backing Python scripts. No Railway proxy. No sub-agents.
The skill is the harness; GLM is the worker.

```
Claude Code (host)
    │
    ├── /write-lesson      ──┐
    ├── /write-exercise      │  gstack agentic skills
    ├── /write-quiz          │  skills/operator-kit/<skill>/SKILL.md
    ├── /build-site-component│  each skill:
    ├── /scan-repo           │    1. reads stage context
    ├── /quality-check       │    2. calls Z.ai GLM via embedded bash
    └── /find-citations    ──┘    3. writes output to stage output/
                                  4. validates / reports
                                  │
                                  ▼
                        api.z.ai/api/coding/paas/v4
                        Authorization: Bearer $ZHIPUAI_API_KEY
                                  │
                          ┌───────┴────────┐
                        GLM-5.1        GLM-4.7-Flash
                        GLM-5          GLM-4.5-Air
                        GLM-4.7        (per skill)
```

## Environment Variables

| Var | Where it lives | Value |
|-----|---------------|-------|
| `ZHIPUAI_API_KEY` | `.env` (gitignored) | Z.ai API key |
| `ZAI_BASE_URL` | `.env` (gitignored) | `https://api.z.ai/api/coding/paas/v4` |

No proxy vars. No Railway. Auth is simple `Bearer $ZHIPUAI_API_KEY` — no JWT signing.

## Confirmed Working Models

| Model | Concurrency | Use in skills |
|-------|-------------|---------------|
| GLM-5.1 | 10 | `/write-lesson`, `/write-exercise`, `/write-quiz` |
| GLM-5 | 2 | `/build-site-component` |
| GLM-4.7 | 2 | `/quality-check` |
| GLM-4.6 | 3 | `/write-lesson` overflow |
| GLM-4.5 | 10 | general content |
| GLM-4.5-Air | 5 | `/find-citations` |
| GLM-4.7-Flash | 1 | `/scan-repo` (fast traversal) |

Note: GLM-5.1, GLM-5, GLM-4.5 are reasoning models — use `max_tokens` ≥ 500.

## Agentic Skill Pattern

Each SKILL.md follows this structure:

```
## When to invoke
[trigger conditions]

## Chain
Step 1 — Read context
  [what files to load, what state to check]

Step 2 — Call GLM
  [embedded bash block: openai SDK call with model, system prompt, user prompt]

Step 3 — Write output
  [where to write, what format, naming convention]

Step 4 — Validate + report
  [what to check, what to surface to Claude Code]
```

The embedded bash block pattern (shared across all skills):

```bash
python3 - <<'PYEOF'
import os, json, sys
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
)

response = client.chat.completions.create(
    model="GLM-5.1",          # override per skill
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT},
    ],
    max_tokens=4000,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
PYEOF
```

## Skill → Model Assignments

| Skill | Function | GLM Model | Active |
|-------|----------|-----------|--------|
| `/write-lesson` | Draft lesson docs + outlines | GLM-5.1 | Stage 01 |
| `/write-quiz` | Quiz banks (FSRS-ready) | GLM-5.1 | Stage 04 |
| `/write-exercise` | Exercise specs | GLM-5.1 | Stage 03 |
| `/build-site-component` | Site components, Helix impl | GLM-5 | Stage 05 |
| `/scan-repo` | Read-only codebase traversal | GLM-4.7-Flash | Stage 01+ |
| `/quality-check` | Curriculum audit, gap detection | GLM-4.7 | Stage 09 |
| `/find-citations` | Gap-fill research, citation finding | GLM-4.5-Air | Stage 01+ |

## What Was Scrapped

| Item | Replaced By |
|------|-------------|
| Railway proxy (BlueSkyGTM/openai-proxy) | Direct Z.ai call from skill |
| Zhipu JWT signing | Simple Bearer token |
| `open.bigmodel.cn` endpoint | `api.z.ai/api/coding/paas/v4` |
| Per-skill `run.py` Python scripts | Embedded bash in SKILL.md |
| Named operator-kit agents (Lyra, Echo, Newton, Hypatia) | Function-named agentic skills |
| `PROXY_URL` / `PROXY_KEY` env vars | `ZAI_BASE_URL` (one var) |
| `vault/index.js` (proxy code) | Archived — no longer needed |

## Build Order

1. ✅ Confirmed Z.ai endpoint + auth working
2. ✅ Updated `.env` (ZAI_BASE_URL, removed proxy vars)
3. ✅ Updated this plan
4. Scaffold `skills/operator-kit/<skill>/SKILL.md` for all 7 skills
5. Update `references/runtime-guide.md` (remove proxy section, add Z.ai note)
6. Smoke-test each skill before Stage 01
7. Archive `vault/index.js`
