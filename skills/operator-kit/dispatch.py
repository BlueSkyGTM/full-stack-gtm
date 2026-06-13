"""
Operator-kit batch dispatcher — parallel GLM lesson outline generation.

Reads stages/01-gtm-skeleton/output/manifest.json, processes all rows where
status == "pending" via ThreadPoolExecutor, writes output files, and updates
manifest status in-place.

Usage:
  python3 skills/operator-kit/dispatch.py                   # all pending rows
  python3 skills/operator-kit/dispatch.py --workers 8       # concurrency (default 4)
  python3 skills/operator-kit/dispatch.py --dry-run         # preview only, no API calls
  python3 skills/operator-kit/dispatch.py --limit 10        # first N pending rows only
  python3 skills/operator-kit/dispatch.py --stage 01        # stage filter (default 01)
  python3 skills/operator-kit/dispatch.py --retry-failed    # re-queue failed rows

Sequential ~2.5-8 hrs for 498 lessons. With 4 workers: ~40-120 min.
With 8 workers: ~20-60 min. Z.ai rate limits apply — back off on 429.

Governed maze: each job gets a distilled ≤500-token system prompt from the brief,
not the full repo context. GLM receives task + brief extract. Nothing else.
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

# ── Config ──────────────────────────────────────────────────────────────────

MANIFEST_PATH = Path("stages/01-gtm-skeleton/output/manifest.json")
BRIEF_PATH    = Path("stages/00-c-agent-setup/output/agent-briefs/lyra-content-brief.md")
OUTPUT_DIR    = Path("stages/01-gtm-skeleton/output/lessons")
TOPIC_MAP     = Path("stages/00-b-gtm-content-mapping/output/gtm-topic-map.md")

BRIEF_SECTIONS = [
    "## Format: Six-Beat Lesson Structure",
    "## GTM Redirect Rules",
    "## Tone Rules",
    "## What NOT To Do",
    "## Code Rules",
    "## Learning Objectives Rules",
]

SYSTEM_PROMPT = """You are Lyra, a GTM engineering curriculum author. Rules:
- Peer-to-peer tone. The student is a practitioner, not a student.
- Mechanism before tool. Explain the algorithm/pattern first, then name the tool.
- No marketing claims. "Clay is powerful" is banned. "Clay implements a waterfall" is correct.
- No scaffolded code. All code must run unmodified and produce observable output.
- No objectives starting with "Understand", "Learn", or "Know".
- If a GTM citation is missing, write [CITATION NEEDED — concept: ...] not an invention.
- No section may be omitted from the six-beat structure."""

# ── Brief extraction (governed maze) ────────────────────────────────────────

def extract_brief() -> str:
    """Extract ≤500-token task-relevant sections from lyra-content-brief.md."""
    if not BRIEF_PATH.exists():
        print(f"ERROR: {BRIEF_PATH} missing — run Stage 00-c first")
        sys.exit(1)
    text = BRIEF_PATH.read_text(encoding="utf-8")
    out = []
    for s in BRIEF_SECTIONS:
        m = re.search(re.escape(s) + r"(.*?)(?=\n## |\Z)", text, re.DOTALL)
        if m:
            out.append(s + m.group(1).rstrip())
    # Cap at ~400 tokens (head -120 lines equivalent)
    result = "\n\n".join(out)
    lines = result.split("\n")[:120]
    return "\n".join(lines)


def extract_gtm_cluster(topic: str) -> str:
    """Pull 1-2 relevant lines from gtm-topic-map for the given topic."""
    if not TOPIC_MAP.exists():
        return ""
    text = TOPIC_MAP.read_text(encoding="utf-8")
    lines = [l for l in text.split("\n") if topic.lower()[:10] in l.lower()]
    return "\n".join(lines[:3])


# ── Manifest I/O ─────────────────────────────────────────────────────────────

manifest_lock = threading.Lock()


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found — run Stage 01 outline generation first")
        sys.exit(1)
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(rows: list[dict]) -> None:
    with manifest_lock:
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(MANIFEST_PATH)


def update_row(rows: list[dict], lesson_id: str, status: str, output_file: str = "") -> None:
    with manifest_lock:
        for row in rows:
            if row.get("id") == lesson_id:
                row["status"] = status
                if output_file:
                    row["output_file"] = output_file
                row["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                break
        # Atomic write every update
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(MANIFEST_PATH)


# ── Single lesson job ────────────────────────────────────────────────────────

def run_job(row: dict, brief: str, client: OpenAI, dry_run: bool) -> tuple[str, str]:
    """
    Process one manifest row. Returns (lesson_id, status).
    Status: "done" | "failed" | "skipped"
    """
    lesson_id = row.get("id", "unknown")
    topic     = row.get("topic", row.get("title", "GTM engineering fundamentals"))
    gtm_cluster = extract_gtm_cluster(topic)

    if dry_run:
        print(f"  [DRY-RUN] {lesson_id}: {topic}")
        return lesson_id, "skipped"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9-]", "", topic.lower().replace(" ", "-"))[:60]
    output_file = OUTPUT_DIR / f"{slug}-draft.md"

    user_prompt = f"""Write a lesson OUTLINE for: {topic}

Six-beat structure and rules (follow exactly):
{brief}

GTM cluster context for "Use It" section:
{gtm_cluster if gtm_cluster.strip() else "Use the closest matching GTM application for this AI concept."}

Output: beat headings + 1-2 sentence description per beat. No full prose. Exercise hooks only (easy/medium/hard), not full exercise text."""

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
                # GLM-5.1 is a reasoning model: output flows through reasoning_content
                # first, then content. Budget tokens for both phases.
                max_tokens=6000,
                stream=True,
            )
            chunks = []
            for chunk in response:
                delta = chunk.choices[0].delta
                # reasoning_content = thinking phase (discard); content = final answer
                if delta.content:
                    chunks.append(delta.content)
            result = "".join(chunks).strip()

            if len(result) < 200:
                raise ValueError(f"Output too short ({len(result)} chars) — likely truncated")

            output_file.write_text(result, encoding="utf-8")
            return lesson_id, "done"

        except RateLimitError:
            retries += 1
            wait = 2 ** retries  # exponential backoff: 2s, 4s, 8s
            print(f"  [429] {lesson_id}: rate limited, waiting {wait}s (attempt {retries}/{max_retries})")
            time.sleep(wait)

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"  [FAIL] {lesson_id}: {e}")
                return lesson_id, "failed"
            time.sleep(1)

    return lesson_id, "failed"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Operator-kit batch dispatcher")
    parser.add_argument("--workers",      type=int, default=4,    help="Thread pool size (default 4)")
    parser.add_argument("--dry-run",      action="store_true",    help="Preview only, no API calls")
    parser.add_argument("--limit",        type=int, default=0,    help="Process only first N pending rows")
    parser.add_argument("--stage",        type=str, default="01", help="Stage filter (default 01)")
    parser.add_argument("--retry-failed", action="store_true",    help="Re-queue rows with status=failed")
    args = parser.parse_args()

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ZHIPUAI_API_KEY not set — check .env")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key or "dry-run",
        base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
    )

    print("Loading manifest...")
    rows = load_manifest()

    # Filter rows to process
    target_statuses = {"pending"}
    if args.retry_failed:
        target_statuses.add("failed")

    pending = [r for r in rows if r.get("status") in target_statuses]
    if args.stage:
        pending = [r for r in pending if str(r.get("stage", "")).lstrip("0") == args.stage.lstrip("0")]
    if args.limit:
        pending = pending[:args.limit]

    total = len(pending)
    if total == 0:
        print("No pending rows. Done.")
        return

    print(f"Extracting brief (governed maze)...")
    brief = extract_brief()
    print(f"Brief: {len(brief.split())} tokens extracted from {BRIEF_PATH.name}")

    print(f"\nDispatching {total} lessons | workers={args.workers} | dry_run={args.dry_run}\n")
    start = time.time()
    done_count = failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_job, row, brief, client, args.dry_run): row
            for row in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                lesson_id, status = future.result()
            except Exception as e:
                lesson_id = row.get("id", "unknown")
                status = "failed"
                print(f"  [ERROR] {lesson_id}: {e}")

            if not args.dry_run:
                update_row(rows, lesson_id, status)
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
        print(f"\n  Re-run failed rows: python3 {__file__} --retry-failed")


if __name__ == "__main__":
    main()
