"""
Stage 03 dispatcher — write exercise specs per hybrid lesson.

Reads stages/03-exercise-design/output/manifest.json, loads the Stage 02 hybrid
lesson for each slot, sends to GLM-5.1 with exercise-writing instructions, and
writes the result to stages/03-exercise-design/output/exercise-specs/<phase>/<lesson>/exercises.md.

Usage:
  python3 skills/operator-kit/dispatch-stage03.py --sample 5     # human gate (first 5)
  python3 skills/operator-kit/dispatch-stage03.py --phase 00-setup-and-tooling
  python3 skills/operator-kit/dispatch-stage03.py                 # full run
  python3 skills/operator-kit/dispatch-stage03.py --dry-run
  python3 skills/operator-kit/dispatch-stage03.py --retry-failed

Environment:
  ZHIPUAI_API_KEY must be exported before running:
    export $(grep -v '^#' .env | xargs)

Governed maze: GLM receives only the lesson content + exercise spec rules.
Not the full repo.

Self-correction mechanisms (inherited from Stage 02 dispatcher):
  Global rate pause: any 429 sets a shared threading.Event for 30s.
  Circuit breaker: >30% failure in last 10 jobs → pause 60s + reset window.
  Status file: stages/03-exercise-design/output/status.json updated every 10 completions.
  Pause sentinel: create stages/03-exercise-design/output/.dispatcher-pause to pause.
"""

import os
import sys
import json
import time
import re
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

# ── Config ────────────────────────────────────────────────────────────────────

S2_MANIFEST    = Path("stages/02-lesson-injection/output/manifest.json")
MANIFEST_PATH  = Path("stages/03-exercise-design/output/manifest.json")
OUTPUT_ROOT    = Path("stages/03-exercise-design/output/exercise-specs")
STATUS_PATH    = Path("stages/03-exercise-design/output/status.json")
PAUSE_SENTINEL = Path("stages/03-exercise-design/output/.dispatcher-pause")

SYSTEM_PROMPT = """You are Lyra, a GTM engineering curriculum author writing exercise specs for hybrid AI/GTM lessons.

Your job: given a full hybrid lesson, write 4–6 exercises that test the lesson's learning objectives.

EXERCISE FORMAT (output this exact structure, no deviation):
## Exercises

1. [Easy exercise — reinforces the core concept. Student runs code or modifies code from the lesson. Observable terminal output.]
2. [Easy exercise — second reinforcement, different angle from Exercise 1.]
3. [Medium exercise — applies the concept to a different problem. Requires transferring the idea. No scaffold given.]
4. [Medium exercise — harder transfer. May reference a slightly different dataset or domain.]
5. [Hard exercise — extends or combines with prior knowledge. No scaffold. Produces a persistent artifact.]
6. [Hard exercise — open-ended design or proof. Optional for shorter lessons; required for capstone-adjacent lessons.]

RULES:
- No scaffolded code. Exercises are fully open-ended tasks, not fill-in-the-blank.
- All exercises must be terminal-executable and produce observable output the student can check.
- Every learning objective must be covered by at least one exercise.
- Hard exercises (5–6) must specify what artifact they produce and where it lands.
  Artifact paths: signals/examples/<name>.py, handlers/<name>.py, or outputs/skill-<name>.md
- Do not include the lesson prose, just the ## Exercises section.
- Do not include a copy-paste flag. Verification is artifact-based.
- GTM application: at least one exercise must involve real GTM tooling (Clay, Apollo, enrichment APIs, CRM data).
- Action verbs: Implement, Build, Compute, Compare, Configure, Trace, Design, Extend, Verify, Apply.
  Never "Understand", "Learn", or "Know".
- Minimum 4 exercises, maximum 6. Most lessons land at 5."""

# ── Global rate control (identical to Stage 02 dispatcher) ───────────────────

_global_pause = threading.Event()
_global_pause.set()  # starts in "go" state (SET=go, CLEAR=pause)
_global_pause_lock = threading.Lock()


def global_rate_pause(duration: int = 30) -> None:
    with _global_pause_lock:
        if _global_pause.is_set():  # only pause if currently going
            print(f"  [GLOBAL BACKOFF] Rate limit — all threads pausing {duration}s")
            _global_pause.clear()
            threading.Timer(duration, _global_pause.set).start()


def wait_if_paused() -> None:
    _global_pause.wait()


def wait_if_sentinel() -> None:
    while PAUSE_SENTINEL.exists():
        print(f"  [SENTINEL] {PAUSE_SENTINEL} exists — waiting 10s")
        time.sleep(10)


# ── Manifest I/O ──────────────────────────────────────────────────────────────

manifest_lock = threading.Lock()


def build_manifest() -> list[dict]:
    """Build Stage 03 manifest from Stage 02 done rows."""
    if not S2_MANIFEST.exists():
        print(f"ERROR: {S2_MANIFEST} not found — run Stage 02 first")
        sys.exit(1)
    s2_rows = json.loads(S2_MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for r in s2_rows:
        if r.get("status") != "done":
            continue
        phase   = r.get("phase", "")
        lesson  = r.get("lesson", "")
        lesson_id = f"s3-{phase}-{lesson}"
        rows.append({
            "id":          lesson_id,
            "phase":       phase,
            "lesson":      lesson,
            "topic":       r.get("topic", ""),
            "s2_lesson":   r.get("output_file", ""),
            "status":      "pending",
            "output_file": "",
            "updated_at":  "",
        })
    return rows


def load_or_create_manifest() -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created Stage 03 manifest: {len(rows)} slots from Stage 02 done rows")
    return rows


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


# ── Single exercise job ───────────────────────────────────────────────────────

def run_job(row: dict, client: OpenAI, dry_run: bool) -> tuple[str, str, str]:
    lesson_id   = row.get("id", "unknown")
    topic       = row.get("topic", "")
    phase       = row.get("phase", "")
    lesson      = row.get("lesson", "")
    s2_path     = row.get("s2_lesson", "")

    if dry_run:
        print(f"  [DRY-RUN] {lesson_id}: {topic}")
        return lesson_id, "skipped", ""

    # Read the Stage 02 hybrid lesson
    lesson_text = ""
    if s2_path and Path(s2_path).exists():
        raw = Path(s2_path).read_text(encoding="utf-8", errors="ignore")
        # Strip GTM Redirect Rules block if GLM emitted it (prompt bleed fix)
        raw = re.sub(r'## GTM Redirect Rules.*?(?=\n## |\Z)', '', raw, flags=re.DOTALL)
        # Feed Learning Objectives + The Concept to GLM (enough context, not full lesson)
        # Extract objectives and concept sections; cap at 3000 chars
        objectives_m = re.search(r'## Learning Objectives(.*?)(?=\n## |\Z)', raw, re.DOTALL)
        concept_m    = re.search(r'## The Concept(.*?)(?=\n## |\Z)', raw, re.DOTALL)
        parts = [f"# {topic}"]
        if objectives_m:
            parts.append("## Learning Objectives" + objectives_m.group(1)[:1500])
        if concept_m:
            parts.append("## The Concept" + concept_m.group(1)[:1000])
        lesson_text = "\n\n".join(parts)
    else:
        lesson_text = f"# {topic}\n\n[No hybrid lesson found — write exercises from topic name only]"

    out_dir = OUTPUT_ROOT / phase / lesson
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "exercises.md"

    user_prompt = f"""Write the ## Exercises section for this lesson.

{lesson_text}

Follow the EXERCISE FORMAT exactly. Output only the ## Exercises section — nothing else."""

    retries = 0
    max_retries = 3
    while retries <= max_retries:
        wait_if_sentinel()
        wait_if_paused()
        try:
            response = client.chat.completions.create(
                model="GLM-5.1",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=8000,  # raised from 2000 — exercise specs were truncating
                stream=True,
                timeout=120,
            )
            chunks = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            result = "".join(chunks).strip()

            if len(result) < 100:
                raise ValueError(f"Output too short ({len(result)} chars)")

            # Write header + exercises
            content = f"# Exercises — {topic}\n\n{result}\n"
            tmp = output_file.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(output_file)
            return lesson_id, "done", str(output_file)

        except RateLimitError:
            global_rate_pause(30)
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
    parser = argparse.ArgumentParser(description="Stage 03 exercise design dispatcher")
    parser.add_argument("--workers",      type=int, default=3)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--sample",       type=int, default=0,  help="Run only first N lessons (human gate)")
    parser.add_argument("--phase",        type=str, default="", help="Filter to one phase slug")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Rebuild manifest from Stage 02 done rows")
    args = parser.parse_args()

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ZHIPUAI_API_KEY not set — export $(grep -v '^#' .env | xargs)")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key or "dry-run",
        base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
    )

    if args.rebuild_manifest and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    rows = load_or_create_manifest()

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

    print(f"\nDispatching {total} exercise specs | workers={args.workers} | dry_run={args.dry_run}\n")

    start = time.time()
    done_count = failed_count = 0
    _window: deque[int] = deque(maxlen=10)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, row, client, args.dry_run): row for row in pending}
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
                _window.append(0)
            elif status == "failed":
                failed_count += 1
                _window.append(1)

            elapsed = time.time() - start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            failure_rate = sum(_window) / len(_window) if _window else 0
            print(f"  [{i}/{total}] {lesson_id} -> {status} | {rate:.1f} req/s | ETA {eta/60:.1f}min | fail_rate={failure_rate:.0%}")

            if len(_window) == 10 and failure_rate > 0.30:
                print(f"  [CIRCUIT BREAKER] {failure_rate:.0%} failure in last 10 jobs — pausing all workers 60s")
                _window.clear()
                global_rate_pause(60)

            if i % 10 == 0 and not args.dry_run:
                STATUS_PATH.write_text(json.dumps({
                    "done": done_count,
                    "failed": failed_count,
                    "pending": total - i,
                    "failure_rate": round(failure_rate, 3),
                    "workers": args.workers,
                    "elapsed_min": round(elapsed / 60, 1),
                    "eta_min": round(eta / 60, 1),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, indent=2), encoding="utf-8")

    elapsed = time.time() - start
    print(f"\n-- Summary ----------------------------------")
    print(f"  Done:   {done_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total:  {total} in {elapsed/60:.1f}min")
    if failed_count:
        print(f"\n  Re-run failed: python3 {__file__} --retry-failed")

    if not args.dry_run:
        STATUS_PATH.write_text(json.dumps({
            "done": done_count,
            "failed": failed_count,
            "pending": 0,
            "failure_rate": round(failed_count / total, 3) if total else 0,
            "workers": args.workers,
            "elapsed_min": round(elapsed / 60, 1),
            "eta_min": 0,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished": True,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
