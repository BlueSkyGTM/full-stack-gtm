#!/usr/bin/env python3
"""
Repair quiz.json files to canonical 6-question schema:
  pre, check, check, check, post, post

Strategies (no content creation):
  A: pre,pre,check,check,check,post,post       -> drop Q1 (extra pre)
  B: pre,pre,check,check,check,post,post,post  -> drop Q1 + drop last post
  C: pre,pre,check,check,check,post            -> relabel Q1 as 'post', move to end
  D: pre,pre,check,check,post,post             -> relabel Q1 as 'check'

Skip:
  - pre,pre,post,post,post (5q - needs new check questions)
  - post,post,post,post,post (fully broken)
  - pre,pre,check,check,check,check (needs posts)
  - any already canonical

Also fixes L013 (constant answer key) by rotating options so correct answer
lands at varying indices.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
CANONICAL = ["pre", "check", "check", "check", "post", "post"]

STRATEGY_A = ["pre", "pre", "check", "check", "check", "post", "post"]
STRATEGY_B = ["pre", "pre", "check", "check", "check", "post", "post", "post"]
STRATEGY_C = ["pre", "pre", "check", "check", "check", "post"]
STRATEGY_D = ["pre", "pre", "check", "check", "post", "post"]

SKIP_PATTERNS = [
    ["pre", "pre", "post", "post", "post"],
    ["post", "post", "post", "post", "post"],
    ["pre", "pre", "check", "check", "check", "check"],
]


def fix_answer_key(questions):
    """Rotate options so correct answers land at different indices (0,1,2,3 cycle)."""
    targets = [1, 2, 0, 3, 1, 2]  # vary across 6 questions
    for idx, q in enumerate(questions):
        current = q.get("correct", 0)
        target = targets[idx % len(targets)]
        options = q.get("options", [])
        if not isinstance(options, list) or len(options) < 2:
            continue
        if current == target:
            continue
        # Rotate options to move correct answer to target position
        correct_val = options[current]
        new_options = [o for i, o in enumerate(options) if i != current]
        new_options.insert(target, correct_val)
        q["options"] = new_options
        q["correct"] = target
    return questions


def repair(quiz_path):
    """Load quiz.json, apply structural repair, return (new_questions, strategy_used)."""
    raw = quiz_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    is_dict = isinstance(data, dict)
    questions = data.get("questions", []) if is_dict else data
    stages = [q.get("stage") for q in questions]

    if stages == CANONICAL:
        return None, "already_canonical"

    for pat in SKIP_PATTERNS:
        if stages == pat:
            return None, f"skip:{','.join(pat)}"

    if stages not in (STRATEGY_A, STRATEGY_B, STRATEGY_C, STRATEGY_D):
        return None, f"unknown:{','.join(stages)}"

    qs = [dict(q) for q in questions]  # shallow copy

    if stages == STRATEGY_A:
        # Drop Q1 (second pre)
        qs = [qs[0]] + qs[2:]
        strategy = "A"
    elif stages == STRATEGY_B:
        # Drop Q1 (second pre) + drop last post
        qs = [qs[0]] + qs[2:-1]
        strategy = "B"
    elif stages == STRATEGY_C:
        # Relabel Q1 as post, move to end
        q1 = dict(qs[1])
        q1["stage"] = "post"
        qs = [qs[0]] + qs[2:] + [q1]
        strategy = "C"
    elif stages == STRATEGY_D:
        # Relabel Q1 as check
        qs[1]["stage"] = "check"
        strategy = "D"

    # Verify structure
    new_stages = [q.get("stage") for q in qs]
    if new_stages != CANONICAL:
        return None, f"repair_failed:{','.join(new_stages)}"

    # Fix constant answer key
    all_correct = [q.get("correct", 0) for q in qs]
    if len(set(all_correct)) == 1:
        qs = fix_answer_key(qs)

    if is_dict:
        data["questions"] = qs
        result = data
    else:
        result = qs

    return result, strategy


def commit_lesson(lesson_dir, strategy):
    """Stage quiz.json and commit with lesson-scoped message."""
    quiz_rel = (lesson_dir / "quiz.json").relative_to(ROOT)
    slug = lesson_dir.name
    phase = lesson_dir.parent.name

    subprocess.run(
        ["git", "add", str(quiz_rel)],
        cwd=ROOT,
        check=True,
    )

    msg = f"fix(quiz): normalize schema to canonical 6q — {phase}/{slug} [strategy {strategy}]"
    subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=ROOT,
        check=True,
    )


def main():
    dry_run = "--dry-run" in sys.argv
    phase_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--phase="):
            phase_filter = arg.split("=", 1)[1]

    stats = {"fixed": 0, "skipped": 0, "errors": 0, "already_canonical": 0}
    results = []

    for phase_dir in sorted(PHASES_DIR.iterdir()):
        if not phase_dir.is_dir():
            continue
        if phase_filter and not phase_dir.name.startswith(phase_filter):
            continue
        for lesson_dir in sorted(l for l in phase_dir.iterdir() if l.is_dir()):
            quiz_path = lesson_dir / "quiz.json"
            if not quiz_path.exists():
                continue

            try:
                new_data, strategy = repair(quiz_path)
            except Exception as e:
                print(f"ERROR {lesson_dir.relative_to(ROOT)}: {e}")
                stats["errors"] += 1
                continue

            if new_data is None:
                if strategy == "already_canonical":
                    stats["already_canonical"] += 1
                else:
                    reason = strategy.split(":", 1)[-1]
                    print(f"SKIP  {lesson_dir.relative_to(ROOT)}  [{reason}]")
                    stats["skipped"] += 1
                    results.append(("skip", str(lesson_dir.relative_to(ROOT)), reason))
                continue

            if dry_run:
                print(f"WOULD FIX  {lesson_dir.relative_to(ROOT)}  [{strategy}]")
                stats["fixed"] += 1
                continue

            quiz_path.write_text(
                json.dumps(new_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            commit_lesson(lesson_dir, strategy)
            print(f"FIXED  {lesson_dir.relative_to(ROOT)}  [{strategy}]")
            stats["fixed"] += 1
            results.append(("fixed", str(lesson_dir.relative_to(ROOT)), strategy))

    print()
    print("=" * 60)
    print(f"Already canonical : {stats['already_canonical']}")
    print(f"Fixed             : {stats['fixed']}")
    print(f"Skipped           : {stats['skipped']}")
    print(f"Errors            : {stats['errors']}")
    if dry_run:
        print("(DRY RUN — no files written)")


if __name__ == "__main__":
    main()
