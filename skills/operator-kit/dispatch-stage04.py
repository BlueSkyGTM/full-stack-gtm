"""
Stage 04 dispatcher — generate FSRS quiz banks per phase.

Reads stages/04-quiz-recall/output/manifest.json, loads the Stage 02 hybrid
lesson + Stage 03 exercise spec for each slot, sends to GLM-5.1 with quiz-
writing instructions, and writes FSRS-formatted cards to:
  stages/04-quiz-recall/output/quiz-bank/<phase>/<lesson>/cards.json

Quiz schema per lesson: 6 questions
  pre-q0      — pre-quiz (1 question before reading)
  check-q0    — check question 1 of 3 (during lesson)
  check-q1    — check question 2 of 3
  check-q2    — check question 3 of 3
  post-q0     — post-quiz question 1 of 2 (end of lesson)
  post-q1     — post-quiz question 2 of 2

Usage:
  python3 skills/operator-kit/dispatch-stage04.py --sample 5
  python3 skills/operator-kit/dispatch-stage04.py --phase 01-math-foundations
  python3 skills/operator-kit/dispatch-stage04.py
  python3 skills/operator-kit/dispatch-stage04.py --dry-run
  python3 skills/operator-kit/dispatch-stage04.py --retry-failed

Environment:
  ZHIPUAI_API_KEY must be set (use run.ps1 or export manually).

Governed maze: GLM receives only the lesson objectives + concept section.
Not the full repo or full lesson prose.
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
S3_OUTPUT      = Path("stages/03-exercise-design/output/exercise-specs")
MANIFEST_PATH  = Path("stages/04-quiz-recall/output/manifest.json")
OUTPUT_ROOT    = Path("stages/04-quiz-recall/output/quiz-bank")
STATUS_PATH    = Path("stages/04-quiz-recall/output/status.json")
PAUSE_SENTINEL = Path("stages/04-quiz-recall/output/.dispatcher-pause")

SYSTEM_PROMPT = """You are Lyra, a GTM engineering curriculum author writing active recall quiz questions.

Your job: given a hybrid AI/GTM lesson, write exactly 6 quiz questions in JSON.

OUTPUT FORMAT — return ONLY valid JSON, no prose before or after:
{
  "lesson_id": "<phase>/<lesson>",
  "topic": "<lesson title>",
  "questions": [
    {
      "id": "pre-q0",
      "type": "pre",
      "stem": "<question text>",
      "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A",
      "explanation": "<why this answer is correct, 1-2 sentences>",
      "strand": "ai-engineering",
      "tags": ["<topic slug>"]
    }
  ]
}

QUESTION TYPES (in order, exactly 6):
1. pre-q0    (type: "pre")     — 1 question BEFORE reading. Tests baseline. OK to be wrong.
2. check-q0  (type: "check")   — During lesson. Tests The Concept section.
3. check-q1  (type: "check")   — During lesson. Tests Build It section.
4. check-q2  (type: "check")   — During lesson. Bridges AI concept to GTM application.
5. post-q0   (type: "post")    — End of lesson. Harder. Requires synthesis.
6. post-q1   (type: "post")    — End of lesson. Tests a subtlety or edge case.

STRAND VALUES: "ai-engineering" or "gtm-application"
  - check-q2, post-q1 should often be "gtm-application"
  - pre-q0, check-q0, check-q1, post-q0 should often be "ai-engineering"

RULES:
- 4 choices per question (A/B/C/D). One correct answer.
- Stems must be specific and testable — no "which of the following is true?" stems.
- Wrong choices must be plausible, not obviously silly.
- Never test vocabulary in isolation. Test application and mechanism.
- pre-q0 may test prior knowledge from a prerequisite phase — that is intended.
- Do not include the lesson prose in your output. Only the JSON object."""

# ── Global rate control ───────────────────────────────────────────────────────

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
        rows.append({
            "id":          f"s4-{phase}-{lesson}",
            "phase":       phase,
            "lesson":      lesson,
            "topic":       r.get("topic", ""),
            "s2_lesson":   r.get("output_file", ""),
            "s3_exercises": str(S3_OUTPUT / phase / lesson / "exercises.md"),
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
    print(f"Created Stage 04 manifest: {len(rows)} slots")
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


# ── Single quiz job ───────────────────────────────────────────────────────────

def extract_section(text: str, heading: str, max_chars: int = 1200) -> str:
    m = re.search(re.escape(heading) + r"(.*?)(?=\n## |\Z)", text, re.DOTALL)
    return (heading + m.group(1)[:max_chars]) if m else ""


def run_job(row: dict, client: OpenAI, dry_run: bool) -> tuple[str, str, str]:
    lesson_id = row.get("id", "unknown")
    topic     = row.get("topic", "")
    phase     = row.get("phase", "")
    lesson    = row.get("lesson", "")
    s2_path   = row.get("s2_lesson", "")

    if dry_run:
        print(f"  [DRY-RUN] {lesson_id}: {topic}")
        return lesson_id, "skipped", ""

    # Build context — objectives + concept section (capped)
    context_parts = [f"# {topic}"]
    if s2_path and Path(s2_path).exists():
        raw = Path(s2_path).read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"## GTM Redirect Rules.*?(?=\n## |\Z)", "", raw, flags=re.DOTALL)
        context_parts.append(extract_section(raw, "## Learning Objectives", 1000))
        context_parts.append(extract_section(raw, "## The Problem", 600))
        context_parts.append(extract_section(raw, "## The Concept", 800))
        context_parts.append(extract_section(raw, "## Use It", 500))

    # Include exercise topics for check-q1 alignment
    s3_path = row.get("s3_exercises", "")
    if s3_path and Path(s3_path).exists():
        ex_raw = Path(s3_path).read_text(encoding="utf-8", errors="ignore")
        context_parts.append("## Exercise Hooks (for check-q1 alignment)\n" + ex_raw[:400])

    context = "\n\n".join(p for p in context_parts if p.strip())

    user_prompt = f"""Write 6 quiz questions for this lesson. Return ONLY the JSON object.

Lesson ID: {phase}/{lesson}

{context}"""

    out_dir = OUTPUT_ROOT / phase / lesson
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "cards.json"

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
                max_tokens=8000,  # raised from 2500 — the 40% JSON-parse failures were truncation
                stream=True,
                timeout=120,
            )
            chunks = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            result = "".join(chunks).strip()

            # Strip markdown fences if GLM wrapped in ```json
            result = re.sub(r"^```json\s*", "", result)
            result = re.sub(r"\s*```$", "", result)

            parsed = json.loads(result)
            questions = parsed.get("questions", [])
            if len(questions) != 6:
                raise ValueError(f"Expected 6 questions, got {len(questions)}")

            tmp = output_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(output_file)
            return lesson_id, "done", str(output_file)

        except RateLimitError:
            global_rate_pause(30)
            retries += 1
            wait = 2 ** retries
            print(f"  [429] {lesson_id}: rate limited, waiting {wait}s (attempt {retries}/{max_retries})")
            time.sleep(wait)

        except (json.JSONDecodeError, ValueError) as e:
            retries += 1
            if retries > max_retries:
                print(f"  [FAIL] {lesson_id}: bad JSON — {e}")
                return lesson_id, "failed", ""
            time.sleep(1)

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"  [FAIL] {lesson_id}: {e}")
                return lesson_id, "failed", ""
            time.sleep(1)

    return lesson_id, "failed", ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 04 quiz dispatcher")
    parser.add_argument("--workers",           type=int, default=3)
    parser.add_argument("--dry-run",           action="store_true")
    parser.add_argument("--sample",            type=int, default=0)
    parser.add_argument("--phase",             type=str, default="")
    parser.add_argument("--retry-failed",      action="store_true")
    parser.add_argument("--rebuild-manifest",  action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ZHIPUAI_API_KEY not set — run .\\run.ps1 stage04 or export manually")
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

    print(f"\nDispatching {total} quiz banks | workers={args.workers} | dry_run={args.dry_run}\n")

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
                print(f"  [CIRCUIT BREAKER] {failure_rate:.0%} failure in last 10 — pausing all workers 60s")
                _window.clear()
                global_rate_pause(60)

            if i % 10 == 0 and not args.dry_run:
                STATUS_PATH.write_text(json.dumps({
                    "done": done_count, "failed": failed_count,
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
        print(f"\n  Re-run: python3 {__file__} --retry-failed")

    if not args.dry_run:
        STATUS_PATH.write_text(json.dumps({
            "done": done_count, "failed": failed_count,
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
