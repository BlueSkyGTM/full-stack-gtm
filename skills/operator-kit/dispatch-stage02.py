"""
Stage 02 dispatcher — expand Stage 01 outlines into full hybrid lessons.

Reads stages/02-lesson-injection/output/manifest.json, calls GLM-5.1 per slot
with the Stage 01 outline + GTM context, and writes full hybrid docs/en.md
to stages/02-lesson-injection/output/hybrid-lessons/<phase>/<lesson>/docs/en.md.

Usage:
  python3 skills/operator-kit/dispatch-stage02.py --sample 5     # first 5 only (human gate)
  python3 skills/operator-kit/dispatch-stage02.py --phase 01     # one phase at a time
  python3 skills/operator-kit/dispatch-stage02.py --workers 4    # full run (after approval)
  python3 skills/operator-kit/dispatch-stage02.py --dry-run      # preview only
  python3 skills/operator-kit/dispatch-stage02.py --retry-failed

Environment:
  ZHIPUAI_API_KEY must be exported (not just sourced) before running:
    export $(grep -v '^#' .env | xargs)
  Note: `source .env` alone is insufficient — child Python processes inherit
  exported variables only, not shell-local ones.

Governed maze: GLM-5.1 receives only the brief extract + outline + GTM cluster.
Not the full repo.
"""

import os
import sys
import json
import time
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

MANIFEST_PATH  = Path("stages/02-lesson-injection/output/manifest.json")
BRIEF_PATH     = Path("stages/00-c-agent-setup/output/agent-briefs/lyra-content-brief.md")
GTM_TOPIC_MAP  = Path("stages/00-b-gtm-content-mapping/output/gtm-topic-map.md")
GTM_HANDBOOK   = Path("shared/gtm-handbook-extract.md")
OUTPUT_ROOT    = Path("stages/02-lesson-injection/output/hybrid-lessons")

BRIEF_SECTIONS = [
    "## Format: Six-Beat Lesson Structure",
    "## GTM Redirect Rules",
    "## Tone Rules",
    "## What NOT To Do",
    "## Code Rules",
    "## Learning Objectives Rules",
]

SYSTEM_PROMPT = """You are Lyra, a GTM engineering curriculum author writing full hybrid lessons.

REQUIRED HEADING STRUCTURE (use these exact section headings, in this order, no substitutions):
## Learning Objectives
## The Problem
## The Concept
## Build It
## Use It
## Ship It
## Exercises
## Key Terms

Rules:
- Peer-to-peer tone. The student is a practitioner, not a senior engineer you are impressing.
- Mechanism before tool. Explain the algorithm/pattern first, then name the tool.
- No marketing claims. "Clay is powerful" is banned. "Clay implements a waterfall" is correct.
- No scaffolded code. All code must run unmodified and produce observable output.
- No objectives starting with "Understand", "Learn", or "Know". Use action verbs: build, compute, implement, trace, compare.
- GTM application is woven into the AI concept — not a separate section. Name the AI concept in the first sentence of every GTM beat.
- If a GTM citation is missing, write [CITATION NEEDED — concept: ...] not an invention.
- No section may be omitted.
- In "## The Concept", include exactly one Mermaid diagram when the concept is a sequence, flowchart, decision tree, or pipeline. Use a ```mermaid fenced block. Skip Mermaid only for abstract/metaphor concepts (those get a Tier 1 image in Stage 06).
- End with a ## Sources block listing every GTM claim and its source."""

# ── Brief extraction (governed maze) ─────────────────────────────────────────

def extract_brief() -> str:
    if not BRIEF_PATH.exists():
        print(f"ERROR: {BRIEF_PATH} missing")
        sys.exit(1)
    text = BRIEF_PATH.read_text(encoding="utf-8")
    out = []
    for s in BRIEF_SECTIONS:
        m = re.search(re.escape(s) + r"(.*?)(?=\n## |\Z)", text, re.DOTALL)
        if m:
            out.append(s + m.group(1).rstrip())
    result = "\n\n".join(out)
    return "\n".join(result.split("\n")[:120])


def extract_gtm_context(phase: str, topic: str) -> str:
    """Pull zone row + relevant handbook snippet for this lesson's GTM application."""
    context_parts = []

    if not GTM_TOPIC_MAP.exists():
        print(f"  WARN: {GTM_TOPIC_MAP} missing — GTM cluster context will be absent. Run Stage 00-b to generate it.")
    if not GTM_HANDBOOK.exists():
        print(f"  WARN: {GTM_HANDBOOK} missing — handbook snippet context will be absent.")

    # Zone row from topic map
    if GTM_TOPIC_MAP.exists():
        text = GTM_TOPIC_MAP.read_text(encoding="utf-8")
        # Find the zone row that matches this phase number
        phase_num = re.search(r'\d+', phase)
        if phase_num:
            n = int(phase_num.group())
            pattern = rf'\|\s*{n:02d}\s*\|.*?\n'
            m = re.search(pattern, text)
            if m:
                context_parts.append("Zone table row:\n" + m.group().strip())

    # Handbook snippet: first 60 lines (cluster descriptions)
    if GTM_HANDBOOK.exists():
        lines = GTM_HANDBOOK.read_text(encoding="utf-8", errors="ignore").split("\n")
        # Search for lines containing topic keywords
        keywords = re.sub(r'[^a-zA-Z0-9 ]', '', topic.lower()).split()[:3]
        relevant = [l for l in lines if any(kw in l.lower() for kw in keywords)][:6]
        if relevant:
            context_parts.append("Handbook context:\n" + "\n".join(relevant))

    return "\n\n".join(context_parts) if context_parts else "No GTM cluster context found — use the closest matching GTM application."


# ── Manifest I/O ──────────────────────────────────────────────────────────────

manifest_lock = threading.Lock()


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found")
        sys.exit(1)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def update_row(rows: list[dict], lesson_id: str, status: str, output_file: str = "") -> None:
    with manifest_lock:
        for row in rows:
            if row.get("id") == lesson_id:
                row["status"] = status
                if output_file:
                    row["output_file"] = output_file
                row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                break
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(MANIFEST_PATH)


# ── Single lesson job ─────────────────────────────────────────────────────────

def run_job(row: dict, brief: str, client: OpenAI, dry_run: bool) -> tuple[str, str, str]:
    lesson_id = row.get("id", "unknown")
    topic     = row.get("topic", "GTM engineering fundamentals")
    phase     = row.get("phase", "")
    lesson    = row.get("lesson", "")
    outline_path = row.get("s1_outline", "")

    if dry_run:
        print(f"  [DRY-RUN] {lesson_id}: {topic}")
        return lesson_id, "skipped", ""

    # Read Stage 01 outline
    outline = ""
    if outline_path and Path(outline_path).exists():
        outline = Path(outline_path).read_text(encoding="utf-8", errors="ignore")[:3000]

    # Read existing AI lesson if it exists (for weaving context)
    ai_lesson_path = Path("phases") / phase / lesson / "docs" / "en.md"
    ai_lesson = ""
    if ai_lesson_path.exists():
        ai_lesson = ai_lesson_path.read_text(encoding="utf-8", errors="ignore")[:2000]

    gtm_context = extract_gtm_context(phase, topic)

    # Output path
    out_dir = OUTPUT_ROOT / phase / lesson / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "en.md"

    user_prompt = f"""Write a FULL hybrid lesson for: {topic}

Expand this outline into complete prose for all six beats. Each beat needs:
- 2-4 substantive paragraphs (no bullet padding)
- At least one runnable code block where the beat calls for it
- GTM application woven into the AI concept (not a separate section)

Stage 01 outline (expand this):
{outline if outline else "[No outline — derive from topic and GTM context]"}

GTM context for the "Use It" and "Ship It" beats:
{gtm_context}

Existing AI engineering lesson content (weave into, do not replace):
{ai_lesson if ai_lesson else "[No existing lesson — write the AI strand from scratch]"}

Brief rules (follow exactly):
{brief}

End with:
## Sources
- [every GTM claim with search pointer or [CITATION NEEDED — concept: ...]]"""

    retries = 0
    max_retries = 3
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                model="GLM-5.1",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=8000,  # full lesson needs more room
                stream=True,
            )
            chunks = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            result = "".join(chunks).strip()

            if len(result) < 500:
                raise ValueError(f"Output too short ({len(result)} chars)")

            output_file.write_text(result, encoding="utf-8")
            return lesson_id, "done", str(output_file)

        except RateLimitError:
            retries += 1
            wait = 2 ** retries
            print(f"  [429] {lesson_id}: rate limited, waiting {wait}s (attempt {retries}/{max_retries})")
            time.sleep(wait)

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"  [FAIL] {lesson_id}: {e}")
                return lesson_id, "failed", ""
            time.sleep(1)

    return lesson_id, "failed", ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 02 lesson injection dispatcher")
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--sample",       type=int, default=0,  help="Run only first N lessons (human gate)")
    parser.add_argument("--phase",        type=str, default="", help="Filter to one phase slug")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ZHIPUAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key or "dry-run",
        base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
    )

    rows = load_manifest()

    target_statuses = {"pending"}
    if args.retry_failed:
        target_statuses.add("failed")

    pending = [r for r in rows if r.get("status") in target_statuses]
    if args.phase:
        pending = [r for r in pending if args.phase in r.get("phase", "")]
    if args.sample:
        pending = pending[:args.sample]

    total = len(pending)
    if total == 0:
        print("No pending rows.")
        return

    print(f"Extracting brief (governed maze)...")
    brief = extract_brief()
    print(f"Brief: {len(brief.split())} tokens from {BRIEF_PATH.name}")
    print(f"\nDispatching {total} lessons | workers={args.workers} | dry_run={args.dry_run}\n")

    start = time.time()
    done_count = failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, row, brief, client, args.dry_run): row for row in pending}
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                lesson_id, status, output_file = future.result()
            except Exception as e:
                lesson_id = row.get("id", "unknown")
                status = "failed"
                output_file = ""
                print(f"  [ERROR] {lesson_id}: {e}")

            if not args.dry_run:
                update_row(rows, lesson_id, status, output_file)
            if status == "done":
                done_count += 1
            elif status == "failed":
                failed_count += 1

            elapsed = time.time() - start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{total}] {lesson_id} -> {status} | {rate:.1f} req/s | ETA {eta/60:.1f}min")

    elapsed = time.time() - start
    print(f"\n-- Summary ----------------------------------")
    print(f"  Done:   {done_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total:  {total} in {elapsed/60:.1f}min")
    if failed_count:
        print(f"\n  Re-run failed: python3 {__file__} --retry-failed")


if __name__ == "__main__":
    main()
