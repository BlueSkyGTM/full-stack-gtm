"""
orchestrator.py — the GLM-5.2 Taskmaster (GLM-native overseer).

Replaces the Claude-sub-agent Taskmaster. One GLM ecosystem:
  GLM-5.2 oversees + judges  |  GLM-5.1 writes  |  manifest is the loop.

For each manifest row whose `bucket` needs work (close / complete / regen — set by
reconcile_lessons.py), the orchestrator:
  1. GLM-5.1 Handler writes/completes the lesson (governed maze: brief extract only,
     32K output, finish_reason logged).
  2. GLM-5.2 Taskmaster judges the output against shared/lean-lesson-spec.md
     (the full spec lives in the 5.2 context, NOT the handler prompt) — the two-tier
     gate: structure-complete (blocks write) and ship-ready (blocks publish).
  3. ship_ready -> write + status "done" | structure-only -> write + "done-structure"
     | retryable (<3) -> re-prompt | else -> "failed" with the gap reason.

Re-entrant (ICL): kill it, relaunch it, it reads the manifest and continues.

Usage:
  python skills/operator-kit/orchestrator.py --sample 2        # gate: 2 lessons, watch it work
  python skills/operator-kit/orchestrator.py --bucket close    # one bucket at a time
  python skills/operator-kit/orchestrator.py --workers 5       # full run
  python skills/operator-kit/orchestrator.py --dry-run         # plan only, no API

Env: ZHIPUAI_API_KEY exported (export $(grep -v '^#' .env | xargs))
"""

import os
import sys
import json
import time
import argparse
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

REPO           = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH  = REPO / "stages/02-lesson-injection/output/manifest.json"
SPEC_PATH      = REPO / "shared/lean-lesson-spec.md"
BRIEF_PATH     = REPO / "stages/00-c-agent-setup/output/agent-briefs/lyra-content-brief.md"
OUTPUT_ROOT    = REPO / "stages/02-lesson-injection/output/hybrid-lessons"
STATUS_PATH    = REPO / "stages/04-quiz-recall/output/../../02-lesson-injection/output/orchestrator-status.json"
ENDPOINT       = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")

TASKMASTER_MODEL = "GLM-5.2"   # 1M context, 131K output, max-effort reasoning — the overseer/judge
HANDLER_MODEL    = "GLM-5.1"   # text writer (proven casing); "GLM-5.1V" for vision tasks
HANDLER_MAXTOK   = 32000       # 131K cap available; 32K is generous for a lean lesson
WORK_BUCKETS     = {"close", "complete", "regen"}

# ── Global rate control (shared with the handler tier) ───────────────────────
_global_pause = threading.Event(); _global_pause.set()
_pause_lock = threading.Lock()
manifest_lock = threading.Lock()

def global_rate_pause(seconds: int = 30) -> None:
    with _pause_lock:
        if _global_pause.is_set():
            print(f"  [GLOBAL BACKOFF] pausing {seconds}s")
            _global_pause.clear()
            threading.Timer(seconds, _global_pause.set).start()

def wait_if_paused() -> None:
    _global_pause.wait()

# ── Manifest I/O ─────────────────────────────────────────────────────────────

def load_manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

def update_row(rows, lesson_id, status, bucket=None, gap=None):
    with manifest_lock:
        for r in rows:
            if r.get("id") == lesson_id:
                r["status"] = status
                if bucket is not None: r["bucket"] = bucket
                if gap: r["gap"] = gap
                r["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                break
        tmp = MANIFEST_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(MANIFEST_PATH)

# ── Governed-maze handler prompt (bucket-specific) ───────────────────────────

def brief_extract() -> str:
    if not BRIEF_PATH.exists():
        return "Write a lean fundamentals lesson: mechanism-first prose, runnable code, GTM weave named in the first sentence of Use It."
    # governed maze: first 120 lines only, never the whole repo
    return "\n".join(BRIEF_PATH.read_text(encoding="utf-8").split("\n")[:120])

def handler_prompt(row: dict, bucket: str, existing: str, brief: str) -> tuple[str, str]:
    topic = row.get("topic", "GTM engineering fundamentals")
    system = (
        "You are Lyra, writing a LEAN fundamentals lesson (1,200-1,800 words). "
        "Required sections in order: ## Learning Objectives (3, action verbs) / ## The Problem / "
        "## The Concept (one valid ```mermaid block) / ## Build It (runnable from scratch) / "
        "## Use It (15-30 line RUNNABLE GTM slice; name the AI mechanism in the first sentence) / "
        "## Exercises (2) / ## Key Terms / ## Sources. No ## Ship It. "
        "If a GTM source is missing, write [CITATION NEEDED — concept: ...], never invent one."
    )
    if bucket == "regen":
        user = f"Write the full lean lesson for: {topic}\n\nBrief rules:\n{brief}"
    elif bucket == "close":
        user = (f"Finish this truncated lean lesson for: {topic}. The front half (Concept + Build It) "
                f"is GOOD — preserve it VERBATIM. Continue from the cut: complete ## Use It as a runnable "
                f"GTM slice, then add ## Exercises, ## Key Terms, ## Sources. Output the COMPLETE lesson.\n\n"
                f"EXISTING (preserve, then continue):\n{existing}\n\nBrief rules:\n{brief}")
    else:  # complete
        user = (f"Complete this truncated lean lesson for: {topic}. It was cut during ## Build It. "
                f"Preserve ## The Concept and the good part of ## Build It, finish Build It, then write "
                f"## Use It (runnable GTM slice), ## Exercises, ## Key Terms, ## Sources. Output the COMPLETE lesson.\n\n"
                f"EXISTING (preserve, then continue):\n{existing}\n\nBrief rules:\n{brief}")
    return system, user

# ── GLM-5.1 Handler call ─────────────────────────────────────────────────────

def handler_call(client: OpenAI, system: str, user: str) -> tuple[str, str]:
    """Returns (text, finish_reason)."""
    response = client.chat.completions.create(
        model=HANDLER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=HANDLER_MAXTOK, stream=True, timeout=180,
    )
    chunks, finish = [], "unknown"
    for chunk in response:
        ch = chunk.choices[0]
        if ch.delta and ch.delta.content:
            chunks.append(ch.delta.content)
        if ch.finish_reason:
            finish = ch.finish_reason
    return "".join(chunks).strip(), finish

# ── GLM-5.2 Taskmaster judge (the two-tier gate, reasoning-judged) ───────────

def judge_call(client: OpenAI, lesson_text: str, spec: str, finish_reason: str) -> dict:
    """GLM-5.2 evaluates a lesson against the lean spec. Returns the verdict dict."""
    system = (
        "You are the Taskmaster overseeing a GTM curriculum pipeline. Judge the lesson below "
        "against the LEAN LESSON SPEC. Return ONLY a JSON object, no prose:\n"
        '{"structure_complete": bool, "ship_ready": bool, "retryable": bool, "gap": "one short sentence"}\n'
        "structure_complete = balanced ``` fences, all 9 ## headers in order, each section real, one mermaid in The Concept. "
        "ship_ready = structure_complete AND zero [CITATION NEEDED] AND Use It names the AI mechanism in its first sentence AND Use It has a runnable code block. "
        "retryable = fixable by re-prompting (truncated/missing section); false if fundamentally off-topic. "
        f"The generating call's finish_reason was: {finish_reason} (if 'length', it truncated)."
    )
    user = f"=== LEAN LESSON SPEC ===\n{spec}\n\n=== LESSON TO JUDGE ===\n{lesson_text}"
    resp = client.chat.completions.create(
        model=TASKMASTER_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2000, timeout=180,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Conservative fallback: if the judge output is unparseable, treat as retryable.
        return {"structure_complete": False, "ship_ready": False, "retryable": True,
                "gap": "judge output unparseable"}

# ── One lesson: handler writes, taskmaster judges, decide ────────────────────

def run_job(row: dict, spec: str, brief: str, client: OpenAI, dry_run: bool) -> tuple[str, str, str]:
    lesson_id = row.get("id", "unknown")
    bucket    = row.get("bucket", "regen")
    out_path  = REPO / row["output_file"].replace("\\", "/")
    existing  = out_path.read_text(encoding="utf-8") if out_path.exists() else ""

    if dry_run:
        return lesson_id, f"[plan:{bucket}]", ""

    system, user = handler_prompt(row, bucket, existing, brief)
    for attempt in range(1, 4):
        wait_if_paused()
        try:
            text, finish = handler_call(client, system, user)
            if len(text) < 500:
                raise ValueError(f"handler output too short ({len(text)})")

            verdict = judge_call(client, text, spec, finish)

            if verdict.get("ship_ready"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(".tmp"); tmp.write_text(text, encoding="utf-8"); tmp.replace(out_path)
                return lesson_id, "done", ""
            if verdict.get("structure_complete"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(".tmp"); tmp.write_text(text, encoding="utf-8"); tmp.replace(out_path)
                return lesson_id, "done-structure", verdict.get("gap", "")
            if not verdict.get("retryable"):
                return lesson_id, "failed", verdict.get("gap", "not retryable")
            # retryable: loop again (re-prompt with the same bucket instruction)
        except RateLimitError:
            global_rate_pause(30); time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == 3:
                return lesson_id, "failed", str(e)[:120]
            time.sleep(1)
    return lesson_id, "failed", "exhausted retries"

# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 Taskmaster orchestrator")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--bucket", type=str, default="", help="close|complete|regen")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SPEC_PATH.exists():
        print(f"ERROR: {SPEC_PATH} missing — write the lean spec first"); sys.exit(1)
    spec = SPEC_PATH.read_text(encoding="utf-8")
    brief = brief_extract()

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ZHIPUAI_API_KEY not set"); sys.exit(1)
    client = OpenAI(api_key=api_key or "dry-run", base_url=ENDPOINT)

    rows = load_manifest()
    work = [r for r in rows if r.get("status") == "failed" and r.get("bucket", "regen") in WORK_BUCKETS]
    if args.bucket:
        work = [r for r in work if r.get("bucket") == args.bucket]
    if args.sample:
        work = work[:args.sample]

    total = len(work)
    if total == 0:
        print("No rows need work. Run reconcile_lessons.py first, or all buckets clear."); return

    print(f"GLM-native orchestrator | taskmaster={TASKMASTER_MODEL} handler={HANDLER_MODEL} endpoint={ENDPOINT}")
    print(f"Work: {total} rows | workers={args.workers} | dry_run={args.dry_run}\n")

    start = time.time(); done = struct = failed = 0
    win: deque[int] = deque(maxlen=10)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, r, spec, brief, client, args.dry_run): r for r in work}
        for i, fut in enumerate(as_completed(futs), 1):
            lid, status, gap = fut.result()
            if not args.dry_run and status != "skipped":
                update_row(rows, lid, "done" if status in ("done", "done-structure") else status,
                           gap=gap or None)
            if status == "done": done += 1; win.append(0)
            elif status == "done-structure": struct += 1; win.append(0)
            elif status == "failed": failed += 1; win.append(1)
            fr = sum(win) / len(win) if win else 0
            el = time.time() - start
            rate = i / el if el > 0 else 0
            print(f"  [{i}/{total}] {lid} -> {status} | {rate:.1f}/s | fail={fr:.0%}" + (f" | {gap}" if gap else ""))
            if len(win) == 10 and fr > 0.30:
                print(f"  [CIRCUIT BREAKER] {fr:.0%} fail — pausing 60s"); win.clear(); global_rate_pause(60)
            if i % 10 == 0 and not args.dry_run:
                STATUS_PATH.write_text(json.dumps({"done": done, "done_structure": struct, "failed": failed,
                    "pending": total - i, "failure_rate": round(fr, 3),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2), encoding="utf-8")

    print(f"\n-- Summary --\n  ship-ready done: {done}\n  structure-only:  {struct}\n  failed:          {failed}\n  total: {total} in {(time.time()-start)/60:.1f}min")
    if failed or struct:
        print("  Re-run: python skills/operator-kit/orchestrator.py --workers 5")
    if not args.dry_run:
        STATUS_PATH.write_text(json.dumps({"done": done, "done_structure": struct, "failed": failed,
            "pending": 0, "finished": True,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
