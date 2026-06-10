#!/usr/bin/env python3
"""Variance mold: spread the correct-option position so no quiz has a constant answer key.

Mechanical only — reorders options within a question and updates `correct`. Never
changes question text, option text, explanations, or which option is correct.
Questions whose options reference position/other options (e.g. "all of the above",
"both B and C") are left untouched. See quiz-factory/REFERENCES.md (variance rule).

Usage (repo root):
    python quiz-factory/scripts/balance_answers.py            # dry-run, constant-key quizzes
    python quiz-factory/scripts/balance_answers.py --write     # apply
    python quiz-factory/scripts/balance_answers.py --all       # consider every quiz, not just constant
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASES = ROOT / "phases"

POSITIONAL_RE = re.compile(
    r"(all of the|none of the|both of|both a|both b|both are|both nets|both stay|both rise|"
    r"both depend|both rely|both bypass|both require|both tiers|both debaters|both call|"
    r"both metrics|both shard|both replicate|both integrate|both feed|both produce|"
    r"\bneither\b|\bboth\b|a and b|b and c|i and ii|of the above|answers above|"
    r"options above|all of them)",
    re.IGNORECASE,
)


def _stable_base(lesson: str) -> int:
    return int(hashlib.md5(lesson.encode("utf-8")).hexdigest(), 16) % 4


def _load(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data, data.get("questions", [])
    if isinstance(data, list):
        return data, data
    return data, []


def _is_positional(q: dict) -> bool:
    return any(isinstance(o, str) and POSITIONAL_RE.search(o) for o in q.get("options", []))


def balance_quiz(path: Path, force_all: bool) -> dict | None:
    data, questions = _load(path)
    keys = [q.get("correct") for q in questions if isinstance(q, dict)]
    if not keys:
        return None
    constant = len(set(keys)) == 1
    if not constant and not force_all:
        return None

    lesson = data.get("lesson", path.parent.name) if isinstance(data, dict) else path.parent.name
    base = _stable_base(lesson)

    changes = []
    for qi, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        options = q.get("options")
        correct = q.get("correct")
        if not isinstance(options, list) or not isinstance(correct, int):
            continue
        n = len(options)
        if n < 2 or not (0 <= correct < n):
            continue
        if _is_positional(q):
            continue
        desired = (base + qi) % n
        if desired == correct:
            continue
        before = list(options)
        options[desired], options[correct] = options[correct], options[desired]
        assert sorted(map(str, before)) == sorted(map(str, options)), "content drift"
        q["correct"] = desired
        changes.append((qi, correct, desired))

    new_keys = [q.get("correct") for q in questions if isinstance(q, dict)]
    if len(set(new_keys)) == 1:
        for qi, q in enumerate(questions):
            if not isinstance(q, dict) or _is_positional(q):
                continue
            options = q.get("options")
            correct = q.get("correct")
            if not isinstance(options, list) or not isinstance(correct, int) or len(options) < 2:
                continue
            target = (correct + 1) % len(options)
            options[target], options[correct] = options[correct], options[target]
            q["correct"] = target
            changes.append((qi, correct, target))
            break

    if not changes:
        return None

    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "old_keys": keys,
        "new_keys": [q.get("correct") for q in questions if isinstance(q, dict)],
        "data": data,
        "changes": changes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="apply changes")
    parser.add_argument("--all", action="store_true", help="consider every quiz, not just constant-key")
    args = parser.parse_args(argv)

    quizzes = sorted(PHASES.glob("*/*/quiz.json"))
    results = []
    for path in quizzes:
        try:
            r = balance_quiz(path, args.all)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            return 2
        if r:
            results.append(r)

    for r in results:
        flag = "WRITE" if args.write else "DRY"
        print(f"[{flag}] {r['path']}  {r['old_keys']} -> {r['new_keys']}")
        if args.write:
            Path(ROOT / r["path"]).write_text(
                json.dumps(r["data"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    still_constant = [r["path"] for r in results if len(set(r["new_keys"])) == 1]
    print(f"\n{'wrote' if args.write else 'would change'}: {len(results)} quizzes")
    if still_constant:
        print(f"still constant after pass (all-positional): {len(still_constant)}")
        for p in still_constant:
            print("   ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
